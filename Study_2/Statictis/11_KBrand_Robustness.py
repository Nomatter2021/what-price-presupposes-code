"""
10_KBrand_Robustness.py

Academic Title:
    Robustness Analysis of the KBrand Variable:
    Distributional Stability, Outlier Sensitivity, and Temporal Consistency

Objective:
    This script assesses the statistical robustness of the KBrand variable,
    which is hypothesized to capture brand strength or intangible capital.
    The analysis is performed on non‑Financials sectors and includes:

        1. Descriptive statistics by configuration (Normal, C1–C6).
        2. Mann‑Whitney U tests comparing:
            - Collapse (C1/C6) vs. Sustain (C2)
            - Evolve (C3/C4) vs. Sustain (C2)
        3. Bootstrap confidence intervals (95%) for means and medians.
        4. Sensitivity analysis: repeat tests after removing outliers (IQR method).
        5. Temporal stability: pre‑2020 vs. post‑2020 subsamples.

    The script outputs a full academic report and CSV files with detailed results.

Output Files (saved in data/results/):
    - 10_KBrand_Robustness_Report.txt   : Main report.
    - 10_KBrand_Descriptives.csv        : Descriptive stats per configuration.
    - 10_KBrand_MannWhitney.csv         : Pairwise test results.
    - 10_KBrand_Bootstrap_CI.csv        : Bootstrap CIs for mean/median.
    - 10_KBrand_Sensitivity.csv         : Results after outlier removal.
    - 10_KBrand_Temporal.csv            : Pre/post 2020 comparison.

Dependencies:
    pandas, numpy, scipy, pathlib
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import mannwhitneyu, bootstrap
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_FILE = Path('../data/final_panel.csv')
OUTPUT_DIR = Path('results')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Output files
TXT_REPORT = OUTPUT_DIR / 'report/10_KBrand_Robustness_Report.txt'
CSV_DESCRIPTIVES = OUTPUT_DIR / 'table/10_KBrand_Descriptives.csv'
CSV_MW = OUTPUT_DIR / 'table/10_KBrand_MannWhitney.csv'
CSV_BOOT = OUTPUT_DIR / 'table/10_KBrand_Bootstrap_CI.csv'
CSV_SENS = OUTPUT_DIR / 'table/10_KBrand_Sensitivity.csv'
CSV_TEMP = OUTPUT_DIR / 'table/10_KBrand_Temporal.csv'

RANDOM_SEED = 42
N_BOOT = 1000
CONF_LEVEL = 0.95
SPLIT_YEAR = 2020

# Excluded sectors (Financials)
EXCLUDED_SECTORS = ['Financials_and_Real_Estate', 'Financial']

# Valid configurations
VALID_CONFIGS = ['Normal', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
CRASH_STATES = ['C1', 'C6']
EVOLVE_STATES = ['C3', 'C4']
SUSTAIN_STATE = 'C2'

np.random.seed(RANDOM_SEED)


# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================

def load_and_prepare():
    """Load data, clean, exclude Financials, and keep necessary columns."""
    df = pd.read_csv(DATA_FILE)
    df['period_end'] = pd.to_datetime(df['period_end'])

    # Apply formal gate for Normal state
    if 'Regime_Label' in df.columns:
        df['Configuration'] = np.where(
            df['Regime_Label'] == 'Normal_Regime',
            'Normal',
            df['Configuration']
        )

    # Exclude Financials sector
    if 'Sector' in df.columns:
        df = df[~df['Sector'].isin(EXCLUDED_SECTORS)].copy()

    # Keep only valid configurations
    df = df[df['Configuration'].isin(VALID_CONFIGS)].copy()

    # Ensure KBrand is numeric
    if 'KBrand' not in df.columns:
        raise KeyError("Column 'KBrand' not found in the dataset. Please check the data.")
    df['KBrand'] = pd.to_numeric(df['KBrand'], errors='coerce')

    # Drop rows with missing KBrand
    df = df.dropna(subset=['KBrand', 'Configuration'])

    # Add year for temporal split
    df['Year'] = df['period_end'].dt.year

    return df


def get_group_samples(df, group1_states, group2_states):
    """Extract KBrand values for two groups defined by state lists."""
    vals1 = df[df['Configuration'].isin(group1_states)]['KBrand'].dropna().values
    vals2 = df[df['Configuration'].isin(group2_states)]['KBrand'].dropna().values
    return vals1, vals2


# ============================================================================
# DESCRIPTIVE STATISTICS
# ============================================================================

def compute_descriptives(df):
    """Compute mean, median, std, IQR per configuration."""
    stats_list = []
    for cfg in VALID_CONFIGS:
        vals = df[df['Configuration'] == cfg]['KBrand'].dropna()
        if len(vals) == 0:
            continue
        stats_list.append({
            'Configuration': cfg,
            'N': len(vals),
            'Mean': vals.mean(),
            'Median': vals.median(),
            'Std': vals.std(),
            'Q1': vals.quantile(0.25),
            'Q3': vals.quantile(0.75),
            'Min': vals.min(),
            'Max': vals.max()
        })
    return pd.DataFrame(stats_list)


# ============================================================================
# MANN‑WHITNEY U TESTS
# ============================================================================

def run_mannwhitney(df, comparisons):
    """
    Perform Mann‑Whitney U tests for each comparison.
    comparisons: list of dicts with 'name', 'group1', 'group2'.
    """
    results = []
    for comp in comparisons:
        vals1, vals2 = get_group_samples(df, comp['group1'], comp['group2'])
        if len(vals1) < 3 or len(vals2) < 3:
            u_stat, p_val = np.nan, np.nan
        else:
            u_stat, p_val = mannwhitneyu(vals1, vals2, alternative='two-sided')
        results.append({
            'Comparison': comp['name'],
            'Group1': str(comp['group1']),
            'Group2': str(comp['group2']),
            'N1': len(vals1),
            'N2': len(vals2),
            'Median1': np.median(vals1) if len(vals1) > 0 else np.nan,
            'Median2': np.median(vals2) if len(vals2) > 0 else np.nan,
            'U_statistic': u_stat,
            'P_value': p_val,
            'Significant_05': p_val < 0.05 if not np.isnan(p_val) else False
        })
    return pd.DataFrame(results)


# ============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================================

def bootstrap_ci(data, statistic='mean', n_resamples=N_BOOT, confidence_level=CONF_LEVEL):
    """
    Compute bootstrap confidence interval for a statistic.
    statistic: 'mean' or 'median'
    """
    if len(data) < 5:
        return np.nan, np.nan, np.nan
    if statistic == 'mean':
        def stat_fn(x, axis):
            return np.mean(x, axis=axis)
    elif statistic == 'median':
        def stat_fn(x, axis):
            return np.median(x, axis=axis)
    else:
        raise ValueError("statistic must be 'mean' or 'median'")

    res = bootstrap((data,), stat_fn, n_resamples=n_resamples,
                    confidence_level=confidence_level, method='percentile',
                    random_state=RANDOM_SEED)
    return res.confidence_interval.low, res.confidence_interval.high, res.bootstrap_distribution.mean()


def bootstrap_by_configuration(df):
    """For each configuration, compute bootstrap CI for mean and median."""
    rows = []
    for cfg in VALID_CONFIGS:
        vals = df[df['Configuration'] == cfg]['KBrand'].dropna().values
        if len(vals) < 5:
            continue
        mean_low, mean_high, mean_est = bootstrap_ci(vals, 'mean')
        med_low, med_high, med_est = bootstrap_ci(vals, 'median')
        rows.append({
            'Configuration': cfg,
            'N': len(vals),
            'Mean_Estimate': mean_est,
            'Mean_CI_lower': mean_low,
            'Mean_CI_upper': mean_high,
            'Median_Estimate': med_est,
            'Median_CI_lower': med_low,
            'Median_CI_upper': med_high
        })
    return pd.DataFrame(rows)


# ============================================================================
# SENSITIVITY ANALYSIS (OUTLIER REMOVAL)
# ============================================================================

def remove_outliers_iqr(data, multiplier=1.5):
    """Remove outliers using IQR method."""
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return data[(data >= lower) & (data <= upper)]


def sensitivity_analysis(df):
    """
    Repeat key Mann‑Whitney tests after removing outliers (IQR) from each group.
    """
    # Original comparisons (collapse vs sustain, evolve vs sustain)
    comparisons = [
        {'name': 'Collapse vs Sustain', 'group1': CRASH_STATES, 'group2': [SUSTAIN_STATE]},
        {'name': 'Evolve vs Sustain', 'group1': EVOLVE_STATES, 'group2': [SUSTAIN_STATE]}
    ]
    results = []
    for comp in comparisons:
        vals1_raw, vals2_raw = get_group_samples(df, comp['group1'], comp['group2'])
        if len(vals1_raw) < 5 or len(vals2_raw) < 5:
            continue
        # Remove outliers within each group separately
        vals1_clean = remove_outliers_iqr(vals1_raw)
        vals2_clean = remove_outliers_iqr(vals2_raw)
        # Mann‑Whitney on cleaned data
        u_clean, p_clean = mannwhitneyu(vals1_clean, vals2_clean, alternative='two-sided')
        results.append({
            'Comparison': comp['name'],
            'Original_N1': len(vals1_raw),
            'Original_N2': len(vals2_raw),
            'Outliers_Removed_N1': len(vals1_raw) - len(vals1_clean),
            'Outliers_Removed_N2': len(vals2_raw) - len(vals2_clean),
            'P_value_original': mannwhitneyu(vals1_raw, vals2_raw, alternative='two-sided')[1],
            'P_value_after_outlier_removal': p_clean,
            'Conclusion': 'Robust' if p_clean < 0.05 == (p_clean < 0.05) else 'Sensitive'
        })
    return pd.DataFrame(results)


# ============================================================================
# TEMPORAL STABILITY (PRE‑2020 VS POST‑2020)
# ============================================================================

def temporal_stability(df, split_year=SPLIT_YEAR):
    """Compare KBrand distribution pre‑ and post‑split year within each configuration."""
    df_pre = df[df['Year'] < split_year]
    df_post = df[df['Year'] >= split_year]
    results = []
    for cfg in VALID_CONFIGS:
        vals_pre = df_pre[df_pre['Configuration'] == cfg]['KBrand'].dropna().values
        vals_post = df_post[df_post['Configuration'] == cfg]['KBrand'].dropna().values
        if len(vals_pre) < 5 or len(vals_post) < 5:
            continue
        u_stat, p_val = mannwhitneyu(vals_pre, vals_post, alternative='two-sided')
        results.append({
            'Configuration': cfg,
            'N_pre': len(vals_pre),
            'N_post': len(vals_post),
            'Median_pre': np.median(vals_pre),
            'Median_post': np.median(vals_post),
            'U_statistic': u_stat,
            'P_value': p_val,
            'Significant_change': p_val < 0.05
        })
    return pd.DataFrame(results)


# ============================================================================
# ACADEMIC REPORT
# ============================================================================

def write_report(desc_df, mw_df, boot_df, sens_df, temp_df, f):
    f.write("=" * 120 + "\n")
    f.write("ACADEMIC REPORT: ROBUSTNESS ANALYSIS OF KBrand VARIABLE\n")
    f.write("Descriptive Statistics, Bootstrap CIs, Outlier Sensitivity, Temporal Stability\n")
    f.write("=" * 120 + "\n\n")

    # I. Descriptive
    f.write("I. DESCRIPTIVE STATISTICS (Full Sample, Excluding Financials)\n")
    f.write("-" * 80 + "\n")
    f.write(desc_df.to_string(index=False) + "\n\n")

    # II. Mann‑Whitney Tests
    f.write("II. MANN‑WHITNEY U TESTS (Key Comparisons)\n")
    f.write("-" * 80 + "\n")
    f.write(mw_df.to_string(index=False) + "\n\n")

    # III. Bootstrap Confidence Intervals
    f.write("III. BOOTSTRAP CONFIDENCE INTERVALS (95% Percentile, N=1000)\n")
    f.write("-" * 80 + "\n")
    f.write(boot_df.to_string(index=False) + "\n\n")

    # IV. Sensitivity Analysis (Outlier Removal)
    f.write("IV. SENSITIVITY ANALYSIS – OUTLIER REMOVAL (IQR method)\n")
    f.write("-" * 80 + "\n")
    if not sens_df.empty:
        f.write(sens_df.to_string(index=False) + "\n")
        f.write("\nInterpretation: 'Robust' means the significance conclusion (p<0.05) remains unchanged.\n\n")
    else:
        f.write("   (Insufficient data for sensitivity analysis.)\n\n")

    # V. Temporal Stability
    f.write("V. TEMPORAL STABILITY (Pre‑2020 vs. Post‑2020)\n")
    f.write("-" * 80 + "\n")
    if not temp_df.empty:
        f.write(temp_df.to_string(index=False) + "\n")
        f.write("\nInterpretation: A significant change (p<0.05) indicates time‑varying behaviour.\n\n")
    else:
        f.write("   (Insufficient data for temporal split.)\n\n")

    f.write("=" * 120 + "\n")
    f.write("Full numerical results are available in the accompanying CSV files.\n")
    f.write("=" * 120 + "\n")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("=" * 80)
    print("ROBUSTNESS ANALYSIS FOR KBrand VARIABLE")
    print("Descriptive, Bootstrap, Sensitivity, Temporal Stability")
    print("=" * 80)

    # Load data
    print("\n[1] Loading and preparing data...")
    df = load_and_prepare()
    print(f"    Total observations: {len(df)}")

    # Descriptive statistics
    print("[2] Computing descriptive statistics...")
    desc_df = compute_descriptives(df)
    desc_df.to_csv(CSV_DESCRIPTIVES, index=False)

    # Mann‑Whitney tests
    print("[3] Running Mann‑Whitney U tests...")
    comparisons = [
        {'name': 'Collapse vs Sustain', 'group1': CRASH_STATES, 'group2': [SUSTAIN_STATE]},
        {'name': 'Evolve vs Sustain', 'group1': EVOLVE_STATES, 'group2': [SUSTAIN_STATE]},
        {'name': 'Collapse vs Evolve', 'group1': CRASH_STATES, 'group2': EVOLVE_STATES},
        {'name': 'Normal vs Collapse', 'group1': ['Normal'], 'group2': CRASH_STATES}
    ]
    mw_df = run_mannwhitney(df, comparisons)
    mw_df.to_csv(CSV_MW, index=False)

    # Bootstrap confidence intervals
    print("[4] Computing bootstrap confidence intervals...")
    boot_df = bootstrap_by_configuration(df)
    boot_df.to_csv(CSV_BOOT, index=False)

    # Sensitivity analysis (outlier removal)
    print("[5] Performing sensitivity analysis (outlier removal)...")
    sens_df = sensitivity_analysis(df)
    sens_df.to_csv(CSV_SENS, index=False)

    # Temporal stability
    print("[6] Checking temporal stability (pre/post 2020)...")
    temp_df = temporal_stability(df, split_year=SPLIT_YEAR)
    temp_df.to_csv(CSV_TEMP, index=False)

    # Write report
    with open(TXT_REPORT, 'w', encoding='utf-8') as f:
        write_report(desc_df, mw_df, boot_df, sens_df, temp_df, f)

    print(f"\n✅ Robustness analysis completed successfully.")
    print(f"   Academic report: {TXT_REPORT}")
    print(f"   CSV outputs     : {CSV_DESCRIPTIVES}, {CSV_MW}, {CSV_BOOT}, {CSV_SENS}, {CSV_TEMP}")


if __name__ == "__main__":
    main()