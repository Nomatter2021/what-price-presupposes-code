"""
STEP 10: TIME-ADJUSTED KBRAND ESTIMATION (GENERIC LOGIC)
Calculates KBrand using a universal dynamic logic without hardcoded tickers.
1. Reads Sector Multipliers dynamically from 'survey_config.yaml'.
2. Reads historical Brand Scores from 'data/interbrand_scores.csv' (if available).
3. Applies a common fallback logic based on 'Type' for missing data.
"""

import pandas as pd
import os
import yaml
from datetime import datetime
from pathlib import Path

# ==================== CONFIGURATION ====================
CONFIG_FILE = Path('survey_config.yaml')
DATA_FOLDER = Path('data/raw')
BRAND_SCORES_FILE = Path('data/interbrand_scores.csv')

# Generic fallback scores if external data is missing
DEFAULT_SCORES = {
    'Mature': 70,  # Established market presence
    'Focal': 30    # Emerging/Speculative brand
}

# Fallback multipliers if not defined in YAML
DEFAULT_MULTIPLIERS = {
    'Technology': {'Mature': 0.35, 'Focal': 0.65},
    'Retail': {'Mature': 0.12, 'Focal': 0.35},
    'Services': {'Mature': 0.25, 'Focal': 0.55}
}

def load_brand_scores():
    """Dynamically loads brand scores from an external CSV file."""
    if not BRAND_SCORES_FILE.exists():
        return {}
    
    try:
        # Expected CSV format: Ticker, 2015, 2016, 2017...
        df_scores = pd.read_csv(BRAND_SCORES_FILE, index_col='Ticker')
        return df_scores.to_dict(orient='index')
    except Exception as e:
        print(f"  ⚠️ Error loading {BRAND_SCORES_FILE}: {e}")
        return {}

def get_brand_score(ticker, year, comp_type, external_scores):
    """
    Generic logic to retrieve or extrapolate brand scores.
    Falls back to a default type-based score if ticker is not in the dataset.
    """
    scores = external_scores.get(ticker, {})
    
    # 1. Fallback Logic if no specific data exists
    if not scores or pd.isna(scores):
        return DEFAULT_SCORES.get(comp_type, 20)
    
    # Clean NaN values from the dictionary
    clean_scores = {int(k): float(v) for k, v in scores.items() if pd.notna(v) and str(k).isdigit()}
    if not clean_scores:
        return DEFAULT_SCORES.get(comp_type, 20)

    years = sorted(clean_scores.keys())
    
    # 2. Match exact year
    if year in clean_scores:
        return clean_scores[year]
    
    # 3. Before available data
    if year < years[0]:
        return clean_scores[years[0]]
    
    # 4. Extrapolate future data
    if year > years[-1]:
        if len(years) >= 2:
            last_growth = (clean_scores[years[-1]] - clean_scores[years[-2]]) / (years[-1] - years[-2])
        else:
            last_growth = 0
        return max(0, min(100, clean_scores[years[-1]] + last_growth * (year - years[-1])))
    
    # 5. Linear interpolation
    before = max(y for y in years if y < year)
    after = min(y for y in years if y > year)
    interpolated = clean_scores[before] + (clean_scores[after] - clean_scores[before]) * (year - before) / (after - before)
    return round(interpolated, 1)

def calculate_kbrand(row, sector, comp_type, base_mult, external_scores):
    """Universal KBrand Formula execution."""
    ticker = row.get('Ticker', 'Unknown')
    
    # Flexibly find revenue column
    revenue_col = next((c for c in row.index if 'revenue' in str(c).lower()), None)
    if not revenue_col or pd.isna(row[revenue_col]) or pd.isna(row.get('period_end')):
        return None
        
    revenue = row[revenue_col]
    period_end = pd.to_datetime(row['period_end'])
    year = period_end.year
    
    brand_score = get_brand_score(ticker, year, comp_type, external_scores)
    brand_factor = brand_score / 100
    
    kbrand = revenue * base_mult * brand_factor
    return round(kbrand, 2)

def process_generic_kbrand():
    print("="*70)
    print("STEP 10: TIME-ADJUSTED KBRAND ESTIMATION (GENERIC PIPELINE)")
    print("="*70)

    if not CONFIG_FILE.exists():
        print(f"❌ Error: {CONFIG_FILE} not found.")
        return

    # Load parameters dynamically
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    external_scores = load_brand_scores()
    if external_scores:
        print(f"✓ Loaded external brand scores for {len(external_scores)} companies.")
    else:
        print("ℹ️ No external brand scores found. Using dynamic generic fallbacks.")

    results = []
    
    for sector_name, sector_info in config.get('sectors', {}).items():
        print(f"\n📑 Sector: {sector_name}")
        
        # Load sector-specific multipliers from YAML (fallback to default if missing)
        sector_multipliers = sector_info.get('multipliers', DEFAULT_MULTIPLIERS.get(sector_name, {}))
        
        for company in sector_info.get('companies', []):
            ticker = company.get('ticker')
            
            # Skip non-active files (Lineage protection)
            if company.get('status') != 'active':
                print(f"  ⏭️ {ticker:6} | Skipped (Status: {company.get('status')})")
                continue

            comp_type = company.get('type', 'Focal').replace('Sa', '')
            file_name = f"{ticker}_raw.csv"
            file_path = DATA_FOLDER / file_name
            
            if not file_path.exists():
                print(f"  ❌ {ticker:6} | FILE NOT FOUND")
                continue
            
            # Get the dynamic multiplier for this type
            base_mult = sector_multipliers.get(comp_type, 0.25)
            
            try:
                df = pd.read_csv(file_path)
                df = df.loc[:, ~df.columns.duplicated()] # Clean
                
                # Temporarily add Ticker for apply function
                df['Ticker'] = ticker
                df['KBrand'] = df.apply(lambda row: calculate_kbrand(row, sector_name, comp_type, base_mult, external_scores), axis=1)
                df.drop(columns=['Ticker'], inplace=True, errors='ignore')
                
                df.to_csv(file_path, index=False, encoding='utf-8-sig')
                
                kb_rows = df['KBrand'].notna().sum()
                print(f"  ✅ {ticker:6} | Computed {kb_rows}/{len(df)} rows (Mult: {base_mult})")
                results.append({'ticker': ticker, 'success': True})
                
            except Exception as e:
                print(f"  ❌ {ticker:6} | ERROR: {str(e)}")
                results.append({'ticker': ticker, 'success': False})

    print("\n" + "="*70)
    print(f"📊 SUMMARY: Processed {sum(1 for r in results if r['success'])} active files.")
    print("="*70)

if __name__ == "__main__":
    process_generic_kbrand()