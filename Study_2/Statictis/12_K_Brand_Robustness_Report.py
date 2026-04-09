"""
12_K_Brand_Classification_Robustness.py (Final Fixed)
Robustness test: change K_Brand scaling (0.25, 0.5, 0.75, 1.0) and re‑classify.
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import logging
from collections import defaultdict
import warnings
import traceback

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG_FILE = Path('../Survey_config.yaml')
RAW_DATA_FOLDER = Path('../data/raw')
BENCHMARK_FOLDER = Path('../data/processed')
OUTPUT_DIR = Path('results')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'report').mkdir(exist_ok=True)
(OUTPUT_DIR / 'table').mkdir(exist_ok=True)

TXT_REPORT = OUTPUT_DIR / 'report/12_K_Brand_Robustness_Report.txt'
CSV_DIST = OUTPUT_DIR / 'table/12_K_Brand_Distribution_Comparison.csv'
CSV_DETAIL = OUTPUT_DIR / 'table/12_K_Brand_Detailed_Changes.csv'

SCALE_FACTORS = [0.25, 0.50, 0.75, 1.00]
TARGET_COL = 'KBrand'  # chính xác theo dữ liệu

# ============================================================================
# HELPER FUNCTIONS (từ 10_Framework_calculate.py)
# ============================================================================

def get_q_period(date):
    if pd.isna(date):
        return None
    return (pd.to_datetime(date) - pd.Timedelta(days=15)).to_period('Q')

def load_benchmark_lookup():
    lookup = {}
    if not BENCHMARK_FOLDER.exists():
        return lookup
    for file_path in BENCHMARK_FOLDER.glob("*_benchmark_median.xlsx"):
        sector = file_path.stem.replace("_benchmark_median", "")
        try:
            df = pd.read_excel(file_path)
            if 's_baseline' in df.columns and 'period_end' in df.columns:
                df['period_end'] = pd.to_datetime(df['period_end'])
                for _, row in df.iterrows():
                    if 'Operating_Margin_median' in df.columns:
                        lookup[(sector, get_q_period(row['period_end']))] = row['Operating_Margin_median']
        except Exception as e:
            logger.error(f"Error loading benchmark for {sector}: {e}")
    return lookup

def get_benchmark_margin(sector, period_end, lookup):
    target_q = get_q_period(period_end)
    if (sector, target_q) in lookup:
        return lookup[(sector, target_q)]
    candidates = [v for k, v in lookup.items() if k[0] == sector and k[1] <= target_q]
    return candidates[-1] if candidates else np.nan

def calculate_framework_metrics(df, sector, benchmark_lookup, scale_factor):
    df = df.copy()
    df['period_end'] = pd.to_datetime(df['period_end'])
    df = df.sort_values('period_end').reset_index(drop=True)

    # Scale KBrand
    if TARGET_COL not in df.columns:
        logger.error(f"Column '{TARGET_COL}' not found. Available: {df.columns.tolist()}")
        return None
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce') * scale_factor

    # Find Revenue column
    rev_col = next((c for c in df.columns if str(c).lower() == 'revenue'), None)
    if rev_col and rev_col != 'Revenue':
        df.rename(columns={rev_col: 'Revenue'}, inplace=True)

    required = ['Revenue', 'market_cap', TARGET_COL]
    if any(c not in df.columns for c in required):
        return None

    df['Revenue'] = pd.to_numeric(df['Revenue'], errors='coerce').fillna(0)
    is_productive = df['Revenue'] > 0

    # Operating Income
    op_inc = next((c for c in df.columns if 'operating' in c.lower() and 'income' in c.lower()), None)
    if not op_inc:
        op_inc = next((c for c in df.columns if 'operatingincome' in c.lower()), None)
    if op_inc:
        df[op_inc] = pd.to_numeric(df[op_inc], errors='coerce').fillna(0)
    else:
        df['Fallback_OpInc'] = 0.0
        op_inc = 'Fallback_OpInc'

    # R&D and SG&A
    rnd = next((c for c in df.columns if 'research' in c.lower()), None)
    sga = next((c for c in df.columns if 'selling' in c.lower()), None)
    if rnd and sga:
        df['Opex_Ratio'] = np.where(is_productive, (df[rnd].fillna(0) + df[sga].fillna(0)) / df['Revenue'], 0.0)
    else:
        df['Opex_Ratio'] = 0.0

    # Margins
    if 'Operating_Margin' not in df.columns:
        df['Operating_Margin'] = np.nan
    cost = df['CostOfRevenue'].fillna(0) if 'CostOfRevenue' in df.columns else 0
    fallback = np.where(is_productive, (df['Revenue'] - cost) / df['Revenue'] - 0.2, np.nan)
    df['Operating_Margin'] = df['Operating_Margin'].fillna(pd.Series(fallback))

    if 'Gross_Margin' not in df.columns:
        df['Gross_Margin'] = np.nan
    use_gross = (df['Operating_Margin'] < 0) & (df['Opex_Ratio'] > 0.30) & (df.get('Gross_Margin', -1) > 0)
    df['Selected_Margin'] = np.where(use_gross, df.get('Gross_Margin', df['Operating_Margin']), df['Operating_Margin'])
    safe_margin = df['Selected_Margin'].clip(lower=None, upper=0.9999)

    # Benchmark
    df['Benchmark_Margin'] = [get_benchmark_margin(sector, d, benchmark_lookup) for d in df['period_end']]

    df['V_Prod_base'] = np.where(is_productive, df['Revenue'] * (1 - safe_margin.clip(lower=0)), 0.0)
    s_base = np.where(df['Benchmark_Margin'].isna(),
                      df['Revenue'] * safe_margin.clip(lower=0),
                      df['Revenue'] * np.minimum(safe_margin.clip(lower=0), df['Benchmark_Margin']))
    df['s_baseline_value'] = np.where(is_productive, s_base, 0.0)
    s_surp = np.where(df['Benchmark_Margin'].isna(), 0.0,
                      df['Revenue'] * np.maximum(0, safe_margin - df['Benchmark_Margin']))
    df['S_Surplus'] = np.where(is_productive, s_surp, df[op_inc].clip(lower=0))
    df['s_total'] = df['s_baseline_value'] + df['S_Surplus']
    df['K_Pi_prime'] = df['market_cap'] - (df['V_Prod_base'] + df['s_total'] + df[TARGET_COL])

    vpb = df['V_Prod_base']
    df['E_star'] = np.where(vpb > 0, (df['market_cap'] - vpb) / vpb, np.nan)
    df['E_0'] = np.where(vpb > 0, df['s_baseline_value'] / vpb, np.nan)
    df['E_1'] = np.where(vpb > 0, df['S_Surplus'] / vpb, np.nan)
    df['E_2'] = np.where(vpb > 0, df[TARGET_COL] / vpb, np.nan)
    df['E_3'] = np.where(vpb > 0, df['K_Pi_prime'] / vpb, np.nan)

    df['K_Pi_prime_lag'] = df['K_Pi_prime'].shift(1)
    df['R_t'] = np.where(df['K_Pi_prime_lag'].notna() & (df['K_Pi_prime_lag'] != 0),
                         df['s_total'] / df['K_Pi_prime_lag'], 0.0)
    df['dK_Pi_prime'] = df['K_Pi_prime'].diff().fillna(0.0)
    df['dK_Pi_prime_pct'] = np.where(df['K_Pi_prime_lag'].notna() & (df['K_Pi_prime_lag'].abs() > 0),
                                     df['dK_Pi_prime'] / df['K_Pi_prime_lag'].abs(), 0.0)
    df['PGR_t'] = np.where(is_productive, df['V_Prod_base'].pct_change().fillna(0.0), 0.0)

    denom = df['dK_Pi_prime'].abs() + df['s_total']
    df['PDI_t'] = np.where((denom != 0) & denom.notna(), df['s_total'] / denom, 0.0)

    df['Gate_C1'] = np.where(df['V_Prod_base'] > 0, (df['K_Pi_prime'] / df['V_Prod_base']) > 0, False)
    df['Gate_C2'] = (df['E_3'] > (df['E_0'] + df['E_1'] + df['E_2'])).astype(bool)
    df['Gate_C3'] = (df['R_t'] < 1.0).astype(bool)
    std_spec = df['Gate_C1'] & df['Gate_C2'] & df['Gate_C3']
    extreme = (df['Revenue'] <= 0) & (df['market_cap'] > 0)
    df['Speculative_Regime'] = np.where(extreme, True, std_spec)

    return df

def classify_state(row):
    try:
        rt = row.get('R_t', np.nan)
        dk = row.get('dK_Pi_prime', np.nan)
        s = row.get('s_total', np.nan)
        dkp = row.get('dK_Pi_prime_pct', np.nan)
        rev = row.get('Revenue', 0.0)
        c1 = row.get('Gate_C1', False)
        c2 = row.get('Gate_C2', False)
        spec = row.get('Speculative_Regime', False)

        if pd.isna(rt): rt = 0.0
        if pd.isna(dk): dk = 0.0
        if pd.isna(s): s = 0.0
        if pd.isna(dkp): dkp = 0.0

        if not ((c1 and c2) or spec):
            return 'Normal'

        if s <= 0:
            if dk <= 0:
                return 'C1' if dkp <= -0.15 else 'C6'
            else:
                return 'C2'
        else:
            if dk > 0:
                return 'C3'
            else:
                if rt >= 0.999 and rev > 0:
                    return 'C5'
                else:
                    return 'C4'
    except Exception:
        return 'Error'

# ============================================================================
# MAIN
# ============================================================================

def main():
    logger.info("=" * 70)
    logger.info("K_Brand CLASSIFICATION ROBUSTNESS TEST (Final)")
    logger.info(f"Scaling factors: {SCALE_FACTORS}")
    logger.info("=" * 70)

    if not CONFIG_FILE.exists():
        logger.error(f"Config file missing: {CONFIG_FILE}")
        return

    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    benchmark_lookup = load_benchmark_lookup()
    logger.info(f"Benchmark lookup size: {len(benchmark_lookup)}")

    # Storage
    config_counts = {f: defaultdict(int) for f in SCALE_FACTORS}
    all_config_lists = {}  # (ticker, factor) -> list of configs

    for sector_name, sector_info in config.get('sectors', {}).items():
        logger.info(f"\nSector: {sector_name}")
        for company in sector_info.get('companies', []):
            if company.get('status') != 'active':
                continue
            ticker = company['ticker']
            file_path = RAW_DATA_FOLDER / sector_name / f"{ticker}_raw.csv"
            if not file_path.exists():
                logger.debug(f"Missing raw data for {ticker}")
                continue

            try:
                df_raw = pd.read_csv(file_path)
                if TARGET_COL not in df_raw.columns:
                    logger.warning(f"Column '{TARGET_COL}' not in {ticker}. Skipping.")
                    continue

                for f in SCALE_FACTORS:
                    df_metrics = calculate_framework_metrics(df_raw, sector_name, benchmark_lookup, f)
                    if df_metrics is None or len(df_metrics) == 0:
                        continue
                    configs = df_metrics.apply(classify_state, axis=1)
                    all_config_lists[(ticker, f)] = configs.tolist()
                    for cfg, cnt in configs.value_counts().items():
                        config_counts[f][cfg] += cnt
            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")
                logger.error(traceback.format_exc())

    if not config_counts[1.00]:
        logger.error("No baseline data. Exiting.")
        return

    # Distribution DataFrame
    all_configs = sorted(set().union(*[list(cnt.keys()) for cnt in config_counts.values()]))
    dist_rows = []
    for f in SCALE_FACTORS:
        total = sum(config_counts[f].values())
        row_dict = {'Factor': f, 'Total_obs': total}
        for cfg in all_configs:
            row_dict[cfg] = config_counts[f].get(cfg, 0)
            row_dict[f'{cfg}_pct'] = 100 * row_dict[cfg] / total if total > 0 else 0
        dist_rows.append(row_dict)
    df_dist = pd.DataFrame(dist_rows)
    df_dist.to_csv(CSV_DIST, index=False)

    # Detailed changes
    change_records = []
    baseline = 1.00
    for (ticker, f), cfg_list in all_config_lists.items():
        if f == baseline:
            continue
        base_list = all_config_lists.get((ticker, baseline))
        if base_list and len(base_list) == len(cfg_list):
            for idx, (base, new) in enumerate(zip(base_list, cfg_list)):
                if base != new:
                    change_records.append({
                        'Ticker': ticker,
                        'Factor': f,
                        'Period_index': idx,
                        'Baseline_Config': base,
                        'New_Config': new
                    })
    df_changes = pd.DataFrame(change_records)
    df_changes.to_csv(CSV_DETAIL, index=False)

    # Write report
    with open(TXT_REPORT, 'w', encoding='utf-8') as report_file:
        report_file.write("=" * 100 + "\n")
        report_file.write("REPORT: K_Brand SCALING ROBUSTNESS TEST\n")
        report_file.write(f"Factors: {SCALE_FACTORS}\n")
        report_file.write("=" * 100 + "\n\n")
        report_file.write("I. DISTRIBUTION OF CONFIGURATIONS\n")
        report_file.write(df_dist.to_string(index=False) + "\n\n")
        report_file.write("II. CHANGES FROM BASELINE (1.00)\n")
        baseline_row = df_dist[df_dist['Factor'] == baseline].iloc[0]
        for f in SCALE_FACTORS:
            if f == baseline:
                continue
            row = df_dist[df_dist['Factor'] == f].iloc[0]
            report_file.write(f"\nFactor {f} vs baseline:\n")
            for cfg in all_configs:
                diff = row[f'{cfg}_pct'] - baseline_row[f'{cfg}_pct']
                report_file.write(f"  {cfg:8}: {diff:+.2f} pp\n")
        report_file.write(f"\nIII. ROW-LEVEL CHANGES: {len(df_changes)} rows changed.\n")
        if not df_changes.empty:
            report_file.write(df_changes.head(20).to_string(index=False))
        report_file.write("\n\n" + "=" * 100 + "\n")

    logger.info(f"✅ Done. Report: {TXT_REPORT}")
    logger.info(f"   Distribution: {CSV_DIST}")
    logger.info(f"   Changes: {CSV_DETAIL}")

if __name__ == "__main__":
    main()