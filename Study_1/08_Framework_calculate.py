"""
STEP 08: SPECULATIVE FRAMEWORK MATHEMATICAL KERNEL
Focus: Strictly computes Labour Theory of Value (LTV) metrics (V_Prod_base, E*, K_Pi', R_t, PDI_t) 
and Formal Gates.
Note: Configuration classification (C1-C6 and Normal) is strictly delegated to Step 09.
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import warnings
from typing import Dict, Tuple, Optional, Any

# Suppress pandas fragmentation warnings for clean console output
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION & PATHS ====================
CONFIG_FILE = Path('survey_config.yaml')
RAW_DATA_FOLDER = Path('data/raw')
PROCESS_FOLDER = Path('data/process')
BENCHMARK_FOLDER = Path('data/processed')

# Ensure output directory exists
PROCESS_FOLDER.mkdir(parents=True, exist_ok=True)

# ==================== HELPER FUNCTIONS ====================

def get_q_period(date: pd.Timestamp) -> Optional[pd.Period]:
    """
    Converts a datetime object to a Quarter period string (e.g., '2023Q1').
    Applies a 15-day backward shift to handle reporting date misalignments.
    """
    if pd.isna(date):
        return None
    return (pd.to_datetime(date) - pd.Timedelta(days=15)).to_period('Q')


def load_benchmark_lookup() -> Dict[Tuple[str, pd.Period], float]:
    """
    Loads the smoothed 3-year rolling benchmark margins from Step 03.
    Returns a dictionary mapped by (Sector, Quarter).
    """
    lookup = {}
    sectors = ['Technology', 'Retail', 'Services']
    
    for sector in sectors:
        csv_path = BENCHMARK_FOLDER / f"{sector}_benchmark_median.csv"
        xlsx_path = BENCHMARK_FOLDER / f"{sector}_benchmark_median.xlsx"
        
        df = None
        if csv_path.exists():
            df = pd.read_csv(csv_path)
        elif xlsx_path.exists():
            df = pd.read_excel(xlsx_path)
            
        if df is not None:
            # Filter for valid benchmark cohorts
            if 'usable_for_benchmark' in df.columns:
                df = df[df['usable_for_benchmark'] == True].copy()
                
            if 'period_end' in df.columns and 'Operating_Margin_median' in df.columns:
                df['period_end'] = pd.to_datetime(df['period_end'])
                for _, row in df.iterrows():
                    if pd.notna(row['Operating_Margin_median']):
                        q_period = get_q_period(row['period_end'])
                        lookup[(sector, q_period)] = row['Operating_Margin_median']
                        
    return lookup


def get_benchmark_margin(sector: str, period_end: pd.Timestamp, lookup: Dict) -> float:
    """
    Retrieves the corresponding benchmark margin. 
    Falls back to the closest past value if the exact quarter is missing.
    """
    target_q = get_q_period(period_end)
    if (sector, target_q) in lookup:
        return lookup[(sector, target_q)]
    
    # Fallback: Find the most recent available historical benchmark
    candidates = [v for k, v in lookup.items() if k[0] == sector and k[1] <= target_q]
    return candidates[-1] if candidates else np.nan


# ==================== CORE MATHEMATICAL ENGINE ====================

def calculate_framework_metrics(df: pd.DataFrame, sector: str, benchmark_lookup: Dict) -> Optional[pd.DataFrame]:
    """
    Calculates core LTV and Speculative Regime metrics strictly according to the paper's methodology.
    """
    df = df.copy()
    df['period_end'] = pd.to_datetime(df['period_end'])
    df = df.sort_values('period_end').reset_index(drop=True)

    # Standardize Revenue column name
    rev_col = next((c for c in df.columns if str(c).lower() == 'revenue'), None)
    if rev_col and rev_col != 'Revenue':
        df.rename(columns={rev_col: 'Revenue'}, inplace=True)

    # Check for mandatory structural columns
    required_cols = ['Revenue', 'market_cap', 'KBrand']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"  ⚠️ Missing required columns: {missing_cols}")
        return None

    # --- 1. Operating Margin & Opex Logic ---
    rnd_col = next((c for c in df.columns if 'research' in c.lower()), None)
    sga_col = next((c for c in df.columns if 'selling' in c.lower()), None)
    
    df['Opex_Ratio'] = 0.0
    if rnd_col and sga_col:
        df['Opex_Ratio'] = (df[rnd_col].fillna(0) + df[sga_col].fillna(0)) / df['Revenue']
    
    if 'Operating_Margin' not in df.columns:
        df['Operating_Margin'] = (df['Revenue'] - df.get('CostOfRevenue', 0)) / df['Revenue'] - 0.2

    # If heavily investing in R&D/SG&A while losing money, fallback to Gross Margin
    use_gross = (df['Operating_Margin'] < 0) & (df['Opex_Ratio'] > 0.30) & (df.get('Gross_Margin', -1) > 0)
    df['Selected_Margin'] = np.where(use_gross, df.get('Gross_Margin', df['Operating_Margin']), df['Operating_Margin'])

    # --- 2. Benchmark & Surplus Calculation ---
    df['Benchmark_Margin'] = [get_benchmark_margin(sector, d, benchmark_lookup) for d in df['period_end']]
    
    # s_baseline: Surplus generated at the socially necessary sector minimum
    df['s_baseline_value'] = np.where(
        df['Benchmark_Margin'].isna(), 
        df['Revenue'] * df['Selected_Margin'].clip(lower=0),
        df['Revenue'] * np.minimum(df['Selected_Margin'].clip(lower=0), df['Benchmark_Margin'])
    )
    
    # S_Surplus: Realized productive superiority beyond the sector floor
    df['S_Surplus'] = np.where(
        df['Benchmark_Margin'].isna(), 
        0.0,
        df['Revenue'] * np.maximum(0, df['Selected_Margin'] - df['Benchmark_Margin'])
    )
    
    df['s_total'] = df['s_baseline_value'] + df['S_Surplus']

    # --- 3. K_Pi' Extraction (The Eleventh Good) ---
    # V_Prod_base captures the accumulated stock of validated labour (c+v)
    df['V_Prod_base'] = df['Revenue'] * (1 - df['Selected_Margin'].clip(lower=0))
    
    # K_Pi': Capitalized claims on uninitiated future cycles (Residual extraction)
    df['K_Pi_prime'] = df['market_cap'] - (df['V_Prod_base'] + df['s_total'] + df['KBrand'])
    
    # --- 4. Dimensionless Ratios (E* Decomposition) ---
    vpb = df['V_Prod_base']
    valid_base = vpb > 0
    
    df['E_star'] = np.where(valid_base, (df['market_cap'] - vpb) / vpb, np.nan)
    df['E_0'] = np.where(valid_base, df['s_baseline_value'] / vpb, np.nan) # Sector baseline
    df['E_1'] = np.where(valid_base, df['S_Surplus'] / vpb, np.nan)        # Superiority
    df['E_2'] = np.where(valid_base, df['KBrand'] / vpb, np.nan)           # Social monopoly
    df['E_3'] = np.where(valid_base, df['K_Pi_prime'] / vpb, np.nan)       # Uninitiated obligation

    # --- 5. System Dynamics & Momentum Indicators ---
    df['K_Pi_prime_lag'] = df['K_Pi_prime'].shift(1)
    
    # R_t (Absorption Ratio): Productive capacity to discharge accumulated obligation
    # Strict forced 0.0 for initial periods to prevent math anomalies
    df['R_t'] = np.where(
        df['K_Pi_prime_lag'].notna() & (df['K_Pi_prime_lag'] != 0), 
        df['s_total'] / df['K_Pi_prime_lag'], 
        0.0  
    )
    
    # T_t: Approximate periods required to discharge obligation
    df['T_t'] = np.where(df['R_t'] > 1e-6, 1 / df['R_t'], np.inf)
    
    # dK: Net movement of the obligation stock
    df['dK_Pi_prime'] = df['K_Pi_prime'].diff().fillna(0.0) 
    
    df['dK_Pi_prime_pct'] = np.where(
        df['K_Pi_prime_lag'].notna() & (df['K_Pi_prime_lag'].abs() > 0), 
        df['dK_Pi_prime'] / df['K_Pi_prime_lag'].abs(), 
        0.0 
    )
    
    # PDI_t (Productive Discharge Index): Measures the source of K_Pi' change
    denominator = df['dK_Pi_prime'].abs() + df['s_total']
    df['PDI_t'] = np.where(
        (denominator != 0) & denominator.notna(), 
        df['s_total'] / denominator, 
        0.0
    )
    
    # PGR_t (Productive Growth Rate)
    df['PGR_t'] = df['V_Prod_base'].pct_change().fillna(0.0)

    # --- 6. Formal Gates: Identify Macro Regime (Normal vs Speculative) ---
    # Condition 1: Uninitiated obligation exceeds verified productive base
    df['Gate_C1'] = (df['K_Pi_prime'] > df['V_Prod_base']).astype(bool)
    
    # Condition 2: Future expectation dominates all history-grounded components
    df['Gate_C2'] = (df['E_3'] > (df['E_0'] + df['E_1'] + df['E_2'])).astype(bool)
    
    # Condition 3: Surplus cannot discharge obligation at the rate it accumulates
    df['Gate_C3'] = (df['R_t'] < 1.0).astype(bool)
    
    # Regime Trigger
    df['Speculative_Regime'] = df['Gate_C1'] & df['Gate_C2'] & df['Gate_C3']

    return df


# ==================== MAIN EXECUTION ====================

def main() -> None:
    print("="*70)
    print("STEP 08: FRAMEWORK MATHEMATICAL KERNEL (E*, K_Pi', R_t, PDI_t)")
    print("="*70)

    if not CONFIG_FILE.exists():
        print(f"❌ Error: Configuration file not found at {CONFIG_FILE}")
        return

    benchmark_lookup = load_benchmark_lookup()
    
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    processed_count = 0
    
    for sector_name, sector_info in config.get('sectors', {}).items():
        print(f"\n📑 Processing Sector: {sector_name}")
        
        for company in sector_info.get('companies', []):
            ticker = company.get('ticker')
            
            if company.get('status') != 'active':
                continue

            fname = f"{ticker}_raw.csv"
            fpath = RAW_DATA_FOLDER / fname
            
            if fpath.exists():
                try:
                    df_raw = pd.read_csv(fpath)
                    df_out = calculate_framework_metrics(df_raw, sector_name, benchmark_lookup)
                    
                    if df_out is not None:
                        out_path = PROCESS_FOLDER / fname
                        df_out.to_csv(out_path, index=False, encoding='utf-8-sig')
                        print(f"  ✅ {ticker:6} | Math kernel computed and saved.")
                        processed_count += 1
                    else:
                        print(f"  ❌ {ticker:6} | Error: Math execution failed (Missing core columns).")
                except Exception as e:
                    print(f"  ❌ {ticker:6} | Unexpected Error: {str(e)}")
            else:
                print(f"  ❌ {ticker:6} | Source file not found: {fpath}")

    print("\n" + "="*70)
    print(f"🎯 KERNEL COMPLETE! Successfully processed mathematics for {processed_count} entities.")
    print("="*70)

if __name__ == "__main__":
    main()
