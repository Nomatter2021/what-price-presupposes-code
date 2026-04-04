"""
STEP 07: TIME-ADJUSTED KBRAND ESTIMATION (GENERIC LOGIC)
Calculates KBrand (Social Monopoly Pricing Power) dynamically.
1. Reads Sector Multipliers from 'survey_config.yaml'.
2. Ingests historical Brand Scores from external sources if available.
3. Applies a robust fallback logic based on firm 'Type' for private/unrated entities.
"""

import pandas as pd
import yaml
import logging
from pathlib import Path
from typing import Dict, Optional, Any

# ==================== CONFIGURATION ====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

CONFIG_FILE = Path('survey_config.yaml')
DATA_FOLDER = Path('data/raw')
BRAND_SCORES_FILE = Path('data/interbrand_scores.csv')

# Structural fallback parameters for unrated entities
DEFAULT_SCORES = {
    'Mature': 70,  # Established market presence
    'Focal': 30    # Emerging/Speculative brand
}

DEFAULT_MULTIPLIERS = {
    'Technology': {'Mature': 0.35, 'Focal': 0.65},
    'Retail': {'Mature': 0.12, 'Focal': 0.35},
    'Services': {'Mature': 0.25, 'Focal': 0.55}
}

# ==================== ESTIMATION LOGIC ====================

def load_brand_scores() -> Dict[str, Dict[int, float]]:
    """Loads external brand evaluations (e.g., ISO 10668/20671 compliant data)."""
    if not BRAND_SCORES_FILE.exists():
        return {}
    try:
        df_scores = pd.read_csv(BRAND_SCORES_FILE, index_col='Ticker')
        return df_scores.to_dict(orient='index')
    except Exception as e:
        logger.warning(f"Could not load external brand scores: {e}")
        return {}

def get_brand_score(ticker: str, year: int, comp_type: str, external_scores: Dict) -> float:
    """
    Retrieves or interpolates brand scores.
    Falls back to structural defaults if entity lacks institutional rating.
    """
    scores = external_scores.get(ticker, {})
    
    if not scores or pd.isna(scores):
        return DEFAULT_SCORES.get(comp_type, 20.0)
    
    # Filter valid numerical scores
    clean_scores = {int(k): float(v) for k, v in scores.items() if pd.notna(v) and str(k).isdigit()}
    if not clean_scores:
        return DEFAULT_SCORES.get(comp_type, 20.0)

    years = sorted(clean_scores.keys())
    
    # Exact match or Boundary constraints
    if year in clean_scores:
        return clean_scores[year]
    if year < years[0]:
        return clean_scores[years[0]]
    if year > years[-1]:
        growth = (clean_scores[years[-1]] - clean_scores[years[-2]]) / (years[-1] - years[-2]) if len(years) >= 2 else 0
        return max(0.0, min(100.0, clean_scores[years[-1]] + growth * (year - years[-1])))
    
    # Linear interpolation for missing intermediary years
    before = max(y for y in years if y < year)
    after = min(y for y in years if y > year)
    interpolated = clean_scores[before] + (clean_scores[after] - clean_scores[before]) * (year - before) / (after - before)
    return round(interpolated, 1)

def calculate_kbrand(row: pd.Series, sector: str, comp_type: str, base_mult: float, external_scores: Dict) -> Optional[float]:
    """Computes KBrand ratio based on the structural formulation."""
    ticker = row.get('Ticker', 'Unknown')
    revenue_col = next((c for c in row.index if 'revenue' in str(c).lower()), None)
    
    if not revenue_col or pd.isna(row[revenue_col]) or pd.isna(row.get('period_end')):
        return None
        
    revenue = float(row[revenue_col])
    year = pd.to_datetime(row['period_end']).year
    
    brand_factor = get_brand_score(ticker, year, comp_type, external_scores) / 100.0
    return round(revenue * base_mult * brand_factor, 2)

# ==================== MAIN EXECUTION ====================

def process_generic_kbrand() -> None:
    logger.info("="*70)
    logger.info("STEP 10: TIME-ADJUSTED KBRAND (SOCIAL MONOPOLY) ESTIMATION")
    logger.info("="*70)

    if not CONFIG_FILE.exists():
        logger.error(f"Config file not found: {CONFIG_FILE}")
        return

    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    external_scores = load_brand_scores()
    
    processed_count = 0
    for sector_name, sector_info in config.get('sectors', {}).items():
        sector_multipliers = sector_info.get('multipliers', DEFAULT_MULTIPLIERS.get(sector_name, {}))
        
        for company in sector_info.get('companies', []):
            if company.get('status') != 'active':
                continue

            ticker = company.get('ticker')
            comp_type = company.get('type', 'Focal').replace('Sa', '')
            file_path = DATA_FOLDER / f"{ticker}_raw.csv"
            
            if not file_path.exists():
                continue
            
            base_mult = sector_multipliers.get(comp_type, 0.25)
            
            try:
                df = pd.read_csv(file_path)
                df = df.loc[:, ~df.columns.duplicated()]
                
                df['Ticker'] = ticker
                df['KBrand'] = df.apply(lambda r: calculate_kbrand(r, sector_name, comp_type, base_mult, external_scores), axis=1)
                df.drop(columns=['Ticker'], inplace=True, errors='ignore')
                
                df.to_csv(file_path, index=False, encoding='utf-8-sig')
                logger.info(f"[{ticker}] KBrand estimated using {comp_type} multiplier ({base_mult}).")
                processed_count += 1
                
            except Exception as e:
                logger.error(f"[{ticker}] Calculation Error: {e}")

    logger.info("="*70)
    logger.info(f"KBRAND ESTIMATION COMPLETE: {processed_count} entities processed.")
    logger.info("="*70)

if __name__ == "__main__":
    process_generic_kbrand()
