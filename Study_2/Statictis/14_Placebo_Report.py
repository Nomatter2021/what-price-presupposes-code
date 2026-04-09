"""
14_Placebo_Robustness.py

Academic Title:
    Placebo Tests for Structural Validity of the Speculative Framework

Objective:
    This script performs three placebo tests to assess whether the observed
    empirical patterns (Markov structure, Kruskal‑Wallis discrimination,
    PDI leading indicator) are genuinely driven by the theoretical content
    of K_Pi_prime and the configuration classification, or are merely
    artifacts of the pipeline.

Tests:
    1. Random K_Pi_prime: Replace K_Pi_prime with random draws from the same
       sector‑year distribution. Recompute E_3, R_t, PDI_t, gates, and
       configurations. Re‑estimate all statistics.
    2. Shuffled configurations: Randomly permute configuration labels within
       each firm's time series (preserving overall proportions). Keep all
       numeric variables unchanged.
    3. Lagged PDI placebo: Test the PDI leading indicator using random lags
       from 2 to 12 quarters (instead of lag 1). Compare significance levels.

Output Files (saved in data/results/):
    - 14_Placebo_Report.txt          : Main comparison table.
    - 14_Placebo_Detailed.csv        : Detailed results per iteration.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import kruskal, mannwhitneyu
import warnings
import logging
from collections import defaultdict
import itertools

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_FILE = Path('../data/final_panel.csv')
OUTPUT_DIR = Path('results')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'report').mkdir(exist_ok=True)
(OUTPUT_DIR / 'table').mkdir(exist_ok=True)

TXT_REPORT = OUTPUT_DIR / 'report/14_Placebo_Report.txt'
CSV_DETAIL = OUTPUT_DIR / 'table/14_Placebo_Detailed.csv'

N_PLACEBO_ITER = 50          # number of iterations for random tests
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

EXCLUDED_SECTORS = ['Financials_and_Real_Estate', 'Financial']
VALID_STATES = ['Normal', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
CRASH_STATES = ['C1', 'C6']
EVOLVE_STATES = ['C3', 'C4']

# ============================================================================
# DATA LOADING AND PREPROCESSING (SAME AS ORIGINAL PIPELINE)
# ============================================================================

def load_and_prepare_data():
    """Load final_panel.csv, apply formal gate, exclude Financials."""
    df = pd.read_csv(DATA_FILE)
    df['period_end'] = pd.to_datetime(df['period_end'])
    if 'Regime_Label' in df.columns:
        df['Configuration'] = np.where(df['Regime_Label'] == 'Normal_Regime', 'Normal', df['Configuration'])
    if 'Sector' in df.columns:
        df = df[~df['Sector'].isin(EXCLUDED_SECTORS)]
    # Keep only valid configurations
    df = df[df['Configuration'].isin(VALID_STATES)].copy()
    # Sort for lag operations
    df = df.sort_values(['Ticker', 'period_end']).reset_index(drop=True)
    return df

# ============================================================================
# FUNCTIONS TO COMPUTE FRAMEWORK METRICS AND CONFIGURATIONS
# (Based on 10_Framework_calculate.py and 11_Classify_configurations.py)
# ============================================================================

def compute_all_metrics(df):
    """
    Compute all framework metrics (E_3, R_t, PDI_t, gates, etc.) from raw data.
    Assumes df contains required columns: Revenue, market_cap, KBrand, etc.
    Returns a DataFrame with added columns.
    """
    df = df.copy()
    # Ensure Revenue exists
    if 'Revenue' not in df.columns:
        # Try to find a revenue-like column
        rev_col = next((c for c in df.columns if str(c).lower() == 'revenue'), None)
        if rev_col:
            df.rename(columns={rev_col: 'Revenue'}, inplace=True)
        else:
            df['Revenue'] = 0.0
    df['Revenue'] = pd.to_numeric(df['Revenue'], errors='coerce').fillna(0)
    is_productive = df['Revenue'] > 0

    # Find Operating Income column (for surplus in M-M')
    op_inc = next((c for c in df.columns if 'operating' in c.lower() and 'income' in c.lower()), None)
    if not op_inc:
        op_inc = next((c for c in df.columns if 'operatingincome' in c.lower()), None)
    if op_inc:
        df[op_inc] = pd.to_numeric(df[op_inc], errors='coerce').fillna(0)
    else:
        df['Fallback_OpInc'] = 0.0
        op_inc = 'Fallback_OpInc'

    # Simplified margin selection (to avoid dependency on benchmark)
    # Use a conservative fixed margin of 0.2 for productive firms
    # This is a simplification; for full accuracy one would need benchmark lookup.
    # For placebo tests, the relative comparison matters more than absolute values.
    margin = 0.2
    df['Selected_Margin'] = margin
    safe_margin = np.clip(df['Selected_Margin'], None, 0.9999)

    df['V_Prod_base'] = np.where(is_productive, df['Revenue'] * (1 - safe_margin), 0.0)
    df['s_baseline_value'] = np.where(is_productive, df['Revenue'] * safe_margin, 0.0)
    df['S_Surplus'] = np.where(is_productive, 0.0, df[op_inc].clip(lower=0))
    df['s_total'] = df['s_baseline_value'] + df['S_Surplus']

    # KBrand column (required)
    if 'KBrand' not in df.columns:
        # Try alternative names
        brand_col = next((c for c in df.columns if 'brand' in c.lower()), None)
        if brand_col:
            df.rename(columns={brand_col: 'KBrand'}, inplace=True)
        else:
            df['KBrand'] = 0.0
    df['KBrand'] = pd.to_numeric(df['KBrand'], errors='coerce').fillna(0)

    # K_Pi_prime (given or computed)
    if 'K_Pi_prime' not in df.columns:
        # Estimate from market cap and components
        df['K_Pi_prime'] = df['market_cap'] - (df['V_Prod_base'] + df['s_total'] + df['KBrand'])
    else:
        df['K_Pi_prime'] = pd.to_numeric(df['K_Pi_prime'], errors='coerce').fillna(0)

    # E ratios
    vpb = df['V_Prod_base'].replace(0, np.nan)
    df['E_0'] = np.where(vpb > 0, df['s_baseline_value'] / vpb, np.nan)
    df['E_1'] = np.where(vpb > 0, df['S_Surplus'] / vpb, np.nan)
    df['E_2'] = np.where(vpb > 0, df['KBrand'] / vpb, np.nan)
    df['E_3'] = np.where(vpb > 0, df['K_Pi_prime'] / vpb, np.nan)

    # Dynamics
    group = df.groupby('Ticker')
    df['K_Pi_prime_lag'] = group['K_Pi_prime'].shift(1)
    df['R_t'] = np.where(df['K_Pi_prime_lag'].notna() & (df['K_Pi_prime_lag'] != 0),
                         df['s_total'] / df['K_Pi_prime_lag'], 0.0)
    df['dK_Pi_prime'] = group['K_Pi_prime'].diff().fillna(0.0)
    df['dK_Pi_prime_pct'] = np.where(df['K_Pi_prime_lag'].notna() & (df['K_Pi_prime_lag'].abs() > 0),
                                     df['dK_Pi_prime'] / df['K_Pi_prime_lag'].abs(), 0.0)
    denom = df['dK_Pi_prime'].abs() + df['s_total']
    df['PDI_t'] = np.where((denom != 0) & denom.notna(), df['s_total'] / denom, 0.0)

    # Gates
    df['Gate_C1'] = np.where(vpb > 0, (df['K_Pi_prime'] / vpb) > 0, False)
    df['Gate_C2'] = (df['E_3'] > (df['E_0'] + df['E_1'] + df['E_2'])).fillna(False)
    df['Gate_C3'] = (df['R_t'] < 1.0).fillna(False)
    standard_spec = df['Gate_C1'] & df['Gate_C2'] & df['Gate_C3']
    extreme_spec = (df['Revenue'] <= 0) & (df['market_cap'] > 0)
    df['Speculative_Regime'] = np.where(extreme_spec, True, standard_spec)

    return df

def classify_configuration(row):
    """Replicate classification logic from 11_Classify_configurations.py."""
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

# ============================================================================
# STATISTICS TO COMPARE
# ============================================================================

def markov_diagonal_mean(df, group_col='Ticker', states=VALID_STATES):
    """Compute mean of diagonal elements of Markov transition matrix (in percent)."""
    df = df.sort_values([group_col, 'period_end'])
    df['next_config'] = df.groupby(group_col)['Configuration'].shift(-1)
    trans = df.dropna(subset=['next_config'])
    trans = trans[trans['Configuration'].isin(states) & trans['next_config'].isin(states)]
    if len(trans) == 0:
        return np.nan
    mat = pd.crosstab(trans['Configuration'], trans['next_config'], normalize='index')
    diag = [mat.loc[s, s] if s in mat.index and s in mat.columns else 0 for s in states]
    return np.mean(diag) * 100

def kruskal_wallis_h(df, var='E_3', group_col='Configuration'):
    """Return H-statistic for Kruskal-Wallis test of var across groups."""
    groups = [df[df[group_col] == cfg][var].dropna().values for cfg in VALID_STATES if cfg in df[group_col].unique()]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        return np.nan
    h, _ = kruskal(*groups)
    return h

def pdi_leading_pvalue(df, group_col='Ticker', lag=1):
    """
    Test whether PDI_t is lower before a crash (C1/C6) than before safe states.
    Uses Mann‑Whitney U test on C3/C4 quarters.
    Returns p-value.
    """
    df = df.copy()
    df['is_crash'] = df['Configuration'].isin(CRASH_STATES).astype(int)
    df['crash_next'] = df.groupby(group_col)['is_crash'].shift(-lag)
    c3c4 = df[df['Configuration'].isin(EVOLVE_STATES)].dropna(subset=['PDI_t', 'crash_next'])
    if len(c3c4) == 0:
        return np.nan
    crash_pdi = c3c4[c3c4['crash_next'] == 1]['PDI_t']
    safe_pdi = c3c4[c3c4['crash_next'] == 0]['PDI_t']
    if len(crash_pdi) < 3 or len(safe_pdi) < 3:
        return np.nan
    _, p = mannwhitneyu(crash_pdi, safe_pdi, alternative='two-sided')
    return p

def compute_all_stats(df):
    """Convenience function to compute the three statistics on a given DataFrame."""
    # Ensure Configuration is set (if not already)
    if 'Configuration' not in df.columns:
        # Compute it from metrics
        df_metrics = compute_all_metrics(df)
        df_metrics['Configuration'] = df_metrics.apply(classify_configuration, axis=1)
        df = df_metrics
    diag_mean = markov_diagonal_mean(df)
    kw_h = kruskal_wallis_h(df, var='E_3')
    pdi_p = pdi_leading_pvalue(df, lag=1)
    return {'diag_mean': diag_mean, 'kw_h': kw_h, 'pdi_p': pdi_p}

# ============================================================================
# PLACEBO TEST 1: RANDOM K_Pi_prime
# ============================================================================

def random_k_pi_prime_placebo(df_orig, n_iter=N_PLACEBO_ITER):
    """
    For each iteration, replace K_Pi_prime with random draws from sector‑year
    distribution, recompute all metrics and configurations, then compute stats.
    """
    results = []
    # Pre‑compute sector‑year groups for random draws
    df_orig = df_orig.copy()
    # Ensure we have sector and year
    if 'Sector' not in df_orig.columns:
        # Use a dummy sector
        df_orig['Sector'] = 'All'
    df_orig['Year'] = df_orig['period_end'].dt.year
    # Group by Sector and Year to get distributions
    groups = df_orig.groupby(['Sector', 'Year'])['K_Pi_prime']
    group_dict = {key: group.dropna().values for key, group in groups}

    for i in range(n_iter):
        logger.info(f"Placebo Test 1: iteration {i+1}/{n_iter}")
        df_rand = df_orig.copy()
        # For each row, draw a random K_Pi_prime from its sector‑year group
        def random_k(row):
            key = (row['Sector'], row['Year'])
            if key in group_dict and len(group_dict[key]) > 0:
                return np.random.choice(group_dict[key])
            else:
                return row['K_Pi_prime']  # fallback
        df_rand['K_Pi_prime'] = df_rand.apply(random_k, axis=1)
        # Recompute all metrics and configurations
        df_metrics = compute_all_metrics(df_rand)
        df_metrics['Configuration'] = df_metrics.apply(classify_configuration, axis=1)
        stats = compute_all_stats(df_metrics)
        stats['iteration'] = i
        results.append(stats)
    return pd.DataFrame(results)

# ============================================================================
# PLACEBO TEST 2: SHUFFLED CONFIGURATIONS
# ============================================================================

def shuffled_configurations_placebo(df_orig, n_iter=N_PLACEBO_ITER):
    """
    Keep all numeric variables unchanged. For each firm, randomly permute
    the configuration labels across its time series (preserving proportions).
    Then compute the same statistics.
    """
    results = []
    for i in range(n_iter):
        logger.info(f"Placebo Test 2: iteration {i+1}/{n_iter}")
        df_shuff = df_orig.copy()
        # Group by Ticker
        for ticker, group in df_orig.groupby('Ticker'):
            cfg_series = group['Configuration'].values
            # Permute the labels
            permuted = np.random.permutation(cfg_series)
            df_shuff.loc[group.index, 'Configuration'] = permuted
        # Compute stats directly on shuffled configs (numeric vars unchanged)
        stats = compute_all_stats(df_shuff)
        stats['iteration'] = i
        results.append(stats)
    return pd.DataFrame(results)

# ============================================================================
# PLACEBO TEST 3: LAGGED PDI
# ============================================================================

def lagged_pdi_placebo(df_orig, lags=range(2, 13)):
    """
    Compute p-value for PDI leading indicator using different lags.
    No randomisation, just a single pass.
    """
    results = []
    for lag in lags:
        p = pdi_leading_pvalue(df_orig, lag=lag)
        results.append({'lag': lag, 'p_value': p})
    return pd.DataFrame(results)

# ============================================================================
# MAIN
# ============================================================================

def main():
    logger.info("=" * 70)
    logger.info("PLACEBO ROBUSTNESS TESTS")
    logger.info("=" * 70)

    # Load original data
    logger.info("Loading original data...")
    df_orig = load_and_prepare_data()
    # Compute baseline statistics on original data (with original configurations)
    logger.info("Computing baseline statistics...")
    baseline_stats = compute_all_stats(df_orig)
    baseline_stats['type'] = 'Real'
    logger.info(f"Baseline: diag_mean={baseline_stats['diag_mean']:.2f}, kw_h={baseline_stats['kw_h']:.2f}, pdi_p={baseline_stats['pdi_p']:.2e}")

    # Test 1: Random K_Pi_prime
    logger.info("\n--- Test 1: Random K_Pi_prime ---")
    df_test1 = random_k_pi_prime_placebo(df_orig)
    test1_mean = df_test1[['diag_mean', 'kw_h', 'pdi_p']].mean()
    test1_std = df_test1[['diag_mean', 'kw_h', 'pdi_p']].std()
    test1_results = pd.DataFrame({
        'type': ['Random K_Pi_prime'] * len(df_test1),
        'iteration': df_test1['iteration'],
        'diag_mean': df_test1['diag_mean'],
        'kw_h': df_test1['kw_h'],
        'pdi_p': df_test1['pdi_p']
    })

    # Test 2: Shuffled configurations
    logger.info("\n--- Test 2: Shuffled configurations ---")
    df_test2 = shuffled_configurations_placebo(df_orig)
    test2_mean = df_test2[['diag_mean', 'kw_h', 'pdi_p']].mean()
    test2_std = df_test2[['diag_mean', 'kw_h', 'pdi_p']].std()
    test2_results = pd.DataFrame({
        'type': ['Shuffled configs'] * len(df_test2),
        'iteration': df_test2['iteration'],
        'diag_mean': df_test2['diag_mean'],
        'kw_h': df_test2['kw_h'],
        'pdi_p': df_test2['pdi_p']
    })

    # Test 3: Lagged PDI
    logger.info("\n--- Test 3: Lagged PDI ---")
    df_test3 = lagged_pdi_placebo(df_orig)
    test3_results = df_test3.copy()
    test3_results['type'] = 'Lagged PDI'

    # Combine all detailed results for CSV
    all_detailed = pd.concat([
        pd.DataFrame([{'type': 'Real', 'iteration': 0, 'diag_mean': baseline_stats['diag_mean'],
                       'kw_h': baseline_stats['kw_h'], 'pdi_p': baseline_stats['pdi_p']}]),
        test1_results,
        test2_results,
        test3_results.rename(columns={'lag': 'iteration'})  # store lag as iteration for simplicity
    ], ignore_index=True)
    all_detailed.to_csv(CSV_DETAIL, index=False)

    # Build comparison table
    comparison = pd.DataFrame([
        {
            'Test': 'Real (Baseline)',
            'Markov diag mean (%)': f"{baseline_stats['diag_mean']:.2f}",
            'KW H-statistic': f"{baseline_stats['kw_h']:.2f}",
            'PDI p-value': f"{baseline_stats['pdi_p']:.2e}"
        },
        {
            'Test': 'Random K_Pi_prime (mean ± std)',
            'Markov diag mean (%)': f"{test1_mean['diag_mean']:.2f} ± {test1_std['diag_mean']:.2f}",
            'KW H-statistic': f"{test1_mean['kw_h']:.2f} ± {test1_std['kw_h']:.2f}",
            'PDI p-value': f"{test1_mean['pdi_p']:.2e} ± {test1_std['pdi_p']:.2e}"
        },
        {
            'Test': 'Shuffled configs (mean ± std)',
            'Markov diag mean (%)': f"{test2_mean['diag_mean']:.2f} ± {test2_std['diag_mean']:.2f}",
            'KW H-statistic': f"{test2_mean['kw_h']:.2f} ± {test2_std['kw_h']:.2f}",
            'PDI p-value': f"{test2_mean['pdi_p']:.2e} ± {test2_std['pdi_p']:.2e}"
        }
    ])

    # Add lagged PDI rows (only p-value)
    for _, row in df_test3.iterrows():
        comparison = pd.concat([comparison, pd.DataFrame([{
            'Test': f'PDI lag {int(row["lag"])}',
            'Markov diag mean (%)': 'N/A',
            'KW H-statistic': 'N/A',
            'PDI p-value': f"{row['p_value']:.2e}"
        }])], ignore_index=True)

    # Write report
    with open(TXT_REPORT, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("PLACEBO ROBUSTNESS TESTS – COMPARISON TABLE\n")
        f.write("=" * 100 + "\n\n")
        f.write(comparison.to_string(index=False) + "\n\n")
        f.write("Interpretation:\n")
        f.write("- For a valid structural model, the Real baseline should show\n")
        f.write("  stronger Markov diagonal, higher KW H-statistic, and much\n")
        f.write("  lower PDI p‑value than the placebo tests.\n")
        f.write("- In particular, the PDI p‑value should be significant (p < 0.05)\n")
        f.write("  only for lag = 1, not for larger lags.\n")
        f.write("=" * 100 + "\n")

    logger.info(f"\n✅ Placebo tests completed. Report: {TXT_REPORT}")
    logger.info(f"Detailed results: {CSV_DETAIL}")

if __name__ == "__main__":
    main()