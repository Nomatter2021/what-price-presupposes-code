"""
13_Check_Identity.py

Academic Title:
    Verification of the Fundamental Identity: E* = E0 + E1 + E2 + E3
    (Including All Sectors, Optionally Filtering)

Usage:
    python 13_Check_Identity.py [--exclude_financials] [--tolerance 1e-6]

By default, includes all sectors found in ../data/process/.
Use --exclude_financials to skip Financials_and_Real_Estate and Financial.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import traceback
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
PROCESS_FOLDER = Path('../data/process')
OUTPUT_DIR = Path('results')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'report').mkdir(exist_ok=True)
(OUTPUT_DIR / 'table').mkdir(exist_ok=True)

TXT_REPORT = OUTPUT_DIR / 'report/13_Identity_Check_Report.txt'
CSV_VIOLATIONS = OUTPUT_DIR / 'table/13_Identity_Violations.csv'
CSV_SUMMARY = OUTPUT_DIR / 'table/13_Identity_Summary.csv'

TOLERANCE = 1e-6
REQUIRED_COLS = [
    'V_Prod_base', 's_baseline_value', 'S_Surplus', 'KBrand', 'K_Pi_prime', 'market_cap'
]
EXCLUDED_SECTORS = ['Financials_and_Real_Estate', 'Financial']

# ============================================================================
# FUNCTIONS
# ============================================================================

def check_identity(df, tolerance=TOLERANCE):
    df = df.copy()
    df['computed_total'] = (df['V_Prod_base'] + df['s_baseline_value'] + 
                            df['S_Surplus'] + df['KBrand'] + df['K_Pi_prime'])
    df['difference'] = df['market_cap'] - df['computed_total']
    df['relative_error'] = np.where(df['market_cap'].abs() > 0,
                                    df['difference'].abs() / df['market_cap'].abs(),
                                    df['difference'].abs())
    df['is_valid'] = df['relative_error'] <= tolerance
    return df

def compute_e_ratios(df):
    df = df.copy()
    vpb = df['V_Prod_base'].replace(0, np.nan)
    df['E_star'] = np.where(vpb > 0, (df['market_cap'] - vpb) / vpb, np.nan)
    df['E0'] = np.where(vpb > 0, df['s_baseline_value'] / vpb, np.nan)
    df['E1'] = np.where(vpb > 0, df['S_Surplus'] / vpb, np.nan)
    df['E2'] = np.where(vpb > 0, df['KBrand'] / vpb, np.nan)
    df['E3'] = np.where(vpb > 0, df['K_Pi_prime'] / vpb, np.nan)
    df['E_sum'] = df['E0'] + df['E1'] + df['E2'] + df['E3']
    df['E_diff'] = df['E_star'] - df['E_sum']
    return df

def process_company(file_path, sector_name):
    try:
        df = pd.read_csv(file_path)
        missing = [col for col in REQUIRED_COLS if col not in df.columns]
        if missing:
            logger.warning(f"Skipping {file_path.name}: missing columns {missing}")
            return None
        for col in REQUIRED_COLS:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df_clean = df.dropna(subset=REQUIRED_COLS).copy()
        if len(df_clean) == 0:
            return None
        df_checked = check_identity(df_clean)
        df_with_e = compute_e_ratios(df_checked)
        ticker = file_path.stem.replace('_processed', '')
        df_with_e['Ticker'] = ticker
        df_with_e['Sector'] = sector_name
        return df_with_e
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        logger.error(traceback.format_exc())
        return None

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Check identity E* = E0+E1+E2+E3')
    parser.add_argument('--exclude_financials', action='store_true',
                        help='Exclude Financials_and_Real_Estate and Financial sectors')
    parser.add_argument('--tolerance', type=float, default=1e-6,
                        help='Numerical tolerance for relative error')
    args = parser.parse_args()

    tolerance = args.tolerance
    exclude_fin = args.exclude_financials

    logger.info("=" * 70)
    logger.info("IDENTITY CHECK: E* = E0 + E1 + E2 + E3")
    if exclude_fin:
        logger.info("Excluding Financials sectors")
    else:
        logger.info("Including ALL sectors (including Financials)")
    logger.info("=" * 70)

    if not PROCESS_FOLDER.exists():
        logger.error(f"Process folder not found: {PROCESS_FOLDER}")
        return

    all_results = []
    violation_rows = []

    for sector_dir in PROCESS_FOLDER.iterdir():
        if not sector_dir.is_dir():
            continue
        sector_name = sector_dir.name
        if exclude_fin and sector_name in EXCLUDED_SECTORS:
            logger.info(f"Skipping sector (excluded): {sector_name}")
            continue
        logger.info(f"Processing sector: {sector_name}")
        for file_path in sector_dir.glob("*_processed.csv"):
            df_res = process_company(file_path, sector_name)
            if df_res is not None and len(df_res) > 0:
                all_results.append(df_res)
                viol = df_res[~df_res['is_valid']]
                if len(viol) > 0:
                    violation_rows.append(viol)

    if not all_results:
        logger.error("No valid processed files found. Check if processed data exists.")
        return

    df_all = pd.concat(all_results, ignore_index=True)
    df_violations = pd.concat(violation_rows, ignore_index=True) if violation_rows else pd.DataFrame()

    # Per-company summary
    summary = df_all.groupby(['Sector', 'Ticker']).agg(
        total_rows=('is_valid', 'count'),
        valid_rows=('is_valid', 'sum'),
        max_abs_diff=('difference', lambda x: x.abs().max()),
        max_rel_error=('relative_error', 'max'),
        mean_rel_error=('relative_error', 'mean')
    ).reset_index()
    summary['valid_pct'] = 100 * summary['valid_rows'] / summary['total_rows']
    summary['identity_passed'] = summary['max_rel_error'] <= tolerance

    summary.to_csv(CSV_SUMMARY, index=False)
    if not df_violations.empty:
        df_violations.to_csv(CSV_VIOLATIONS, index=False)
        logger.info(f"Violations saved to {CSV_VIOLATIONS}")

    with open(TXT_REPORT, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("ACADEMIC REPORT: VERIFICATION OF THE FUNDAMENTAL IDENTITY\n")
        f.write("E* = E0 + E1 + E2 + E3 (market_cap decomposition)\n")
        if exclude_fin:
            f.write("Excluding Financials sectors\n")
        else:
            f.write("Including ALL sectors (Financials included)\n")
        f.write("=" * 100 + "\n\n")

        f.write("I. GLOBAL STATISTICS\n")
        f.write(f"Total number of rows checked: {len(df_all)}\n")
        f.write(f"Number of rows violating identity: {len(df_violations)}\n")
        f.write(f"Proportion of violations: {100 * len(df_violations) / len(df_all):.4f}%\n\n")

        f.write("II. PER‑COMPANY SUMMARY (first 20 rows)\n")
        f.write(summary.head(20).to_string(index=False) + "\n\n")

        f.write("III. NUMERICAL TOLERANCE\n")
        f.write(f"Tolerance used: {tolerance}\n\n")

        f.write("IV. INTERPRETATION\n")
        if len(df_violations) == 0:
            f.write("✅ The identity holds perfectly (within numerical precision) for all rows.\n")
        else:
            f.write("⚠️ Some rows violate the identity.\n")
            f.write(f"   See {CSV_VIOLATIONS} for details.\n")

        f.write("\n" + "=" * 100 + "\n")

    logger.info(f"✅ Identity check completed. Report: {TXT_REPORT}")
    logger.info(f"   Summary: {CSV_SUMMARY}")

if __name__ == "__main__":
    main()