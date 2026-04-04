"""
STEP 8: SPECULATIVE FRAMEWORK CALCULATION (PURE MATH)
Focus: Strictly calculates core LTV metrics (V_Prod_base, E*, K_Pi', R_t, PDI_t) and Formal Gates.
No classification or labeling happens here.

Input: data/raw/{sector}/{ticker}_raw.csv
Output: data/process/{sector}/{ticker}_processed.csv
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import logging
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

CONFIG_FILE = Path('Survey_config.yaml')
RAW_DATA_FOLDER = Path('data/raw')
PROCESS_FOLDER = Path('data/process')
BENCHMARK_FOLDER = Path('data/processed')

def get_q_period(date):
    if pd.isna(date): return None
    return (pd.to_datetime(date) - pd.Timedelta(days=15)).to_period('Q')

def load_benchmark_lookup():
    lookup = {}
    if not BENCHMARK_FOLDER.exists(): return lookup
    for file_path in BENCHMARK_FOLDER.glob("*_benchmark_median.xlsx"):
        sector = file_path.stem.replace("_benchmark_median", "")
        try:
            df = pd.read_excel(file_path)
            if 's_baseline' in df.columns and 'period_end' in df.columns:
                df['period_end'] = pd.to_datetime(df['period_end'])
                for _, row in df.iterrows():
                    if 'Operating_Margin_median' in df.columns and 'period_end' in df.columns:
                        lookup[(sector, get_q_period(row['period_end']))] = row['Operating_Margin_median']
        except Exception as e:
            logger.error(f"  ⚠️ Error loading benchmark for {sector}: {e}")
    return lookup

def get_benchmark_margin(sector, period_end, lookup):
    target_q = get_q_period(period_end)
    if (sector, target_q) in lookup: return lookup[(sector, target_q)]
    candidates = [v for k, v in lookup.items() if k[0] == sector and k[1] <= target_q]
    return candidates[-1] if candidates else np.nan

def calculate_framework_metrics(df, sector, benchmark_lookup):
    """Compute strictly the mathematical components of the LTV framework."""
    df = df.copy()
    df['period_end'] = pd.to_datetime(df['period_end'])
    df = df.sort_values('period_end').reset_index(drop=True)

    # Đảm bảo cột Revenue có chữ R viết hoa (đồng bộ XBRL)
    rev_col = next((c for c in df.columns if str(c).lower() == 'revenue'), None)
    if rev_col and rev_col != 'Revenue':
        df.rename(columns={rev_col: 'Revenue'}, inplace=True)

    required_cols = ['Revenue', 'market_cap', 'KBrand']
    if not all(col in df.columns for col in required_cols): return None

    # 1. Margin Selection
    rnd = next((c for c in df.columns if 'research' in c.lower()), None)
    sga = next((c for c in df.columns if 'selling' in c.lower()), None)
    df['Opex_Ratio'] = (df[rnd].fillna(0) + df[sga].fillna(0)) / df['Revenue'] if rnd and sga else 0
    if 'Operating_Margin' not in df.columns:
        df['Operating_Margin'] = (df['Revenue'] - df.get('CostOfRevenue', 0)) / df['Revenue'] - 0.2
    use_gross = (df['Operating_Margin'] < 0) & (df['Opex_Ratio'] > 0.30) & (df.get('Gross_Margin', -1) > 0)
    df['Selected_Margin'] = np.where(use_gross, df.get('Gross_Margin', df['Operating_Margin']), df['Operating_Margin'])

    # 2. Base & Surplus Components
    df['Benchmark_Margin'] = [get_benchmark_margin(sector, d, benchmark_lookup) for d in df['period_end']]
    df['V_Prod_base'] = df['Revenue'] * (1 - df['Selected_Margin'].clip(lower=0))
    df['s_baseline_value'] = np.where(df['Benchmark_Margin'].isna(), 
                                      df['Revenue'] * df['Selected_Margin'].clip(lower=0),
                                      df['Revenue'] * np.minimum(df['Selected_Margin'].clip(lower=0), df['Benchmark_Margin']))
    df['S_Surplus'] = np.where(df['Benchmark_Margin'].isna(), 0.0,
                               df['Revenue'] * np.maximum(0, df['Selected_Margin'] - df['Benchmark_Margin']))
    df['s_total'] = df['s_baseline_value'] + df['S_Surplus']

    # 3. K_Pi' Extraction (Strict LTV adherence)
    df['K_Pi_prime'] = df['market_cap'] - (df['V_Prod_base'] + df['s_total'] + df['KBrand'])
    
    # 4. E* Ratios
    vpb = df['V_Prod_base']
    df['E_star'] = np.where(vpb > 0, (df['market_cap'] - vpb) / vpb, np.nan)
    df['E_0'] = np.where(vpb > 0, df['s_baseline_value'] / vpb, np.nan)
    df['E_1'] = np.where(vpb > 0, df['S_Surplus'] / vpb, np.nan)
    df['E_2'] = np.where(vpb > 0, df['KBrand'] / vpb, np.nan)
    df['E_3'] = np.where(vpb > 0, df['K_Pi_prime'] / vpb, np.nan)
    
    # 5. Dynamics & Momentum (Đã ép fill 0.0 cho quý đầu tiên)
    df['K_Pi_prime_lag'] = df['K_Pi_prime'].shift(1)
    
    df['R_t'] = np.where(
        df['K_Pi_prime_lag'].notna() & (df['K_Pi_prime_lag'] != 0), 
        df['s_total'] / df['K_Pi_prime_lag'], 
        0.0  # Mặc định = 0 để qua Gate_C3
    )
    
    df['dK_Pi_prime'] = df['K_Pi_prime'].diff().fillna(0.0)
    
    df['dK_Pi_prime_pct'] = np.where(
        df['K_Pi_prime_lag'].notna() & (df['K_Pi_prime_lag'].abs() > 0), 
        df['dK_Pi_prime'] / df['K_Pi_prime_lag'].abs(), 
        0.0
    )
    
    df['PGR_t'] = df['V_Prod_base'].pct_change().fillna(0.0)
    
    # Công thức PDI_t chuẩn
    denominator = df['dK_Pi_prime'].abs() + df['s_total']
    df['PDI_t'] = np.where(
        (denominator != 0) & denominator.notna(), 
        df['s_total'] / denominator, 
        0.0
    )

    # 6. Formal Gates (Rất quan trọng cho Step Phân loại)
    df['Gate_C1'] = (df['K_Pi_prime'] / df['V_Prod_base']).astype(bool)
    df['Gate_C2'] = (df['E_3'] > (df['E_0'] + df['E_1'] + df['E_2'])).astype(bool)
    df['Gate_C3'] = (df['R_t'] < 1.0).astype(bool)
    
    df['Speculative_Regime'] = df['Gate_C1'] & df['Gate_C2'] & df['Gate_C3']
    
    return df

def main():
    logger.info("="*70)
    logger.info("FRAMEWORK METRICS CALCULATION (PURE MATH)")
    logger.info("="*70)

    if not CONFIG_FILE.exists():
        logger.error(f"❌ Config missing: {CONFIG_FILE}")
        return

    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    benchmark_lookup = load_benchmark_lookup()
    stats = {'processed': 0, 'missing_cols': 0, 'file_not_found': 0}

    for sector_name, sector_info in config.get('sectors', {}).items():
        logger.info(f"\nProcessing Sector: {sector_name}")
        sector_process_dir = PROCESS_FOLDER / sector_name
        sector_process_dir.mkdir(parents=True, exist_ok=True)
        
        for company in sector_info.get('companies', []):
            if company.get('status') != 'active': continue
            
            ticker = company['ticker']
            fpath = RAW_DATA_FOLDER / sector_name / f"{ticker}_raw.csv"
            
            if fpath.exists():
                try:
                    df_raw = pd.read_csv(fpath)
                    df_out = calculate_framework_metrics(df_raw, sector_name, benchmark_lookup)
                    if df_out is not None:
                        df_out.to_csv(sector_process_dir / f"{ticker}_processed.csv", index=False, encoding='utf-8-sig')
                        stats['processed'] += 1
                        logger.debug(f"  ✅ {ticker:6} | Math computed.")
                    else:
                        logger.warning(f"  ⚠️ {ticker:6} | Missing required columns.")
                        stats['missing_cols'] += 1
                except Exception as e:
                     logger.error(f"  ❌ {ticker:6} | Error: {e}")
                     stats['missing_cols'] += 1
            else:
                stats['file_not_found'] += 1
                
    logger.info("\n" + "=" * 70)
    logger.info("CALCULATION COMPLETE")
    logger.info(f"  - Successfully processed : {stats['processed']}")
    logger.info(f"  - Failed (Missing Cols)  : {stats['missing_cols']}")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()