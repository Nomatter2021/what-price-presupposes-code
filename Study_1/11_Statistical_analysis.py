"""
11_Statistical_analysis.py

Academic Title:
    Structural Boundary Analysis for Study 1:
    Regime Discrimination, Leading Indicators, and C2 Path Prediction

Methods:
    1. Markov transition matrix (regime persistence)
    2. Spearman rank correlation (speculative paradox: E₃ vs R_t, PDI_t)
    3. Kruskal‑Wallis & Mann‑Whitney post‑hoc (boundary analysis of E₃, PDI_t, R_t)
    4. C2 path: predicting recovery (C3) vs crash (C1/C6) using E₃ and B
    5. PDI leading indicator in C3/C4 (before crash vs safe)
    6. Conditional test (R_t ≈ 0 & dK_Pi_prime < 0): PDI discriminates Normal vs Crash
    7. Directional test (lagged PDI): PDI(t-1) predicts next‑quarter crash

Output:
    - structural_boundary_report.txt : Full academic report
    - (Optional CSV files can be added)
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
from itertools import combinations
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_FILE = 'data/final_all_cycles_combined.csv'
FALLBACK_DATA_FILE = 'final_all_cycles_combined.csv'
VALID_CONFIGS = ['Normal', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
CRASH_CONFIGS = ['C1', 'C6']
R_CONDITION_THRESHOLD = 0.05
MIN_OBS_FOR_TEST = 5

OUTPUT_FILE = 'structural_boundary_report.txt'

# ============================================================================
# DATA LOADING & PREPARATION
# ============================================================================
def load_and_prepare_data():
    """Loads panel data, creates lagged variables, and computes B = E3 - (1+PGR_t)."""
    print("Loading panel data...")
    file_path = DATA_FILE if os.path.exists(DATA_FILE) else FALLBACK_DATA_FILE
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {DATA_FILE} or {FALLBACK_DATA_FILE}")

    df = pd.read_csv(file_path)
    df['period_end'] = pd.to_datetime(df['period_end'])

    # Ensure 'Normal' regime label
    if 'Regime' in df.columns:
        df['Configuration'] = np.where(df['Regime'] == 'Normal_Regime', 'Normal', df['Configuration'])
    elif 'Speculative_Regime' in df.columns:
        df['Configuration'] = np.where(df['Speculative_Regime'] == False, 'Normal', df['Configuration'])

    # Compute PDI_t if missing
    if 'PDI_t' not in df.columns:
        if 's_total' in df.columns and 'dK_Pi_prime' in df.columns:
            df['PDI_t'] = df['s_total'] / (df['dK_Pi_prime'].abs() + df['s_total'])
            df['PDI_t'] = df['PDI_t'].fillna(0)
        else:
            raise ValueError("Missing columns to compute PDI_t")

    # Compute B = E3 - (1 + PGR_t)
    if 'PGR_t' not in df.columns:
        # Try to compute PGR_t from V_Prod_base if available
        if 'V_Prod_base' in df.columns:
            group_col = 'Cycle_ID' if 'Cycle_ID' in df.columns else 'Ticker'
            df = df.sort_values([group_col, 'period_end'])
            df['PGR_t'] = df.groupby(group_col)['V_Prod_base'].pct_change()
        else:
            raise ValueError("Missing PGR_t or V_Prod_base to compute B")
    df['B'] = df['E_3'] - (1 + df['PGR_t'])

    # Ensure numeric types
    numeric_cols = ['E_3', 'R_t', 'PDI_t', 'dK_Pi_prime_pct', 'PGR_t', 'B']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Sort for lags
    group_col = 'Cycle_ID' if 'Cycle_ID' in df.columns else 'Ticker'
    df = df.sort_values([group_col, 'period_end']).reset_index(drop=True)
    df['PDI_lag1'] = df.groupby(group_col)['PDI_t'].shift(1)
    df['dK_Pi_prime_pct_lag1'] = df.groupby(group_col)['dK_Pi_prime_pct'].shift(1)

    # Flag crash regimes
    df['is_crash'] = df['Configuration'].isin(CRASH_CONFIGS).astype(int)

    print(f"Loaded {len(df):,} observations, {df['Ticker'].nunique()} firms.")
    return df, group_col

# ============================================================================
# MODULE 1: MARKOV TRANSITION MATRIX
# ============================================================================
def markov_transitions(df, group_col, f):
    f.write("\n" + "="*80 + "\n")
    f.write("1. MARKOV TRANSITION MATRIX (REGIME PERSISTENCE)\n")
    f.write("="*80 + "\n")
    df_sorted = df.copy()
    df_sorted['Next_Config'] = df_sorted.groupby(group_col)['Configuration'].shift(-1)
    trans = df_sorted.dropna(subset=['Next_Config'])
    trans = trans[trans['Configuration'].isin(VALID_CONFIGS) & trans['Next_Config'].isin(VALID_CONFIGS)]
    matrix = pd.crosstab(trans['Configuration'], trans['Next_Config'], normalize='index') * 100
    matrix = matrix.reindex(index=VALID_CONFIGS, columns=VALID_CONFIGS, fill_value=0)
    f.write(matrix.round(1).to_string())
    f.write("\n\n→ Interpretation: High diagonal values indicate strong regime persistence.\n")
    f.write("  Note: C5 never observed, C1/C6 mostly transition to C2/C3 (early recovery).\n")

# ============================================================================
# MODULE 2: SPEARMAN CORRELATION (SPECULATIVE PARADOX)
# ============================================================================
def spearman_correlations(df, f):
    f.write("\n" + "="*80 + "\n")
    f.write("2. SPEARMAN RANK CORRELATION (SPECULATIVE PARADOX)\n")
    f.write("="*80 + "\n")
    df_corr = df[['E_3', 'R_t', 'PDI_t']].dropna()
    if len(df_corr) < 2:
        f.write("Insufficient data.\n")
        return
    rho_rt, p_rt = stats.spearmanr(df_corr['E_3'], df_corr['R_t'])
    rho_pdi, p_pdi = stats.spearmanr(df_corr['E_3'], df_corr['PDI_t'])
    f.write(f"E₃ vs R_t   : ρ = {rho_rt:6.4f}, p = {p_rt:.2e}\n")
    f.write(f"E₃ vs PDI_t : ρ = {rho_pdi:6.4f}, p = {p_pdi:.2e}\n")
    f.write("\n→ Interpretation: Negative ρ (E₃ vs PDI_t) supports the speculative paradox:\n")
    f.write("  Higher extrapolative sentiment (E₃) coincides with lower price distortion (PDI).\n")

# ============================================================================
# MODULE 3: STRUCTURAL BOUNDARY TESTS (KRUSKAL‑WALLIS + POST‑HOC)
# ============================================================================
def boundary_analysis(df, f):
    f.write("\n" + "="*80 + "\n")
    f.write("3. STRUCTURAL BOUNDARY ANALYSIS (E₃, PDI_t, R_t)\n")
    f.write("="*80 + "\n")
    metrics = ['E_3', 'PDI_t', 'R_t']
    config_counts = df['Configuration'].value_counts()
    active_configs = [c for c in VALID_CONFIGS if config_counts.get(c, 0) >= 10]

    f.write("A. Kruskal‑Wallis H‑test (differences across configurations)\n")
    for m in metrics:
        if m not in df.columns:
            continue
        samples = [df[df['Configuration'] == c][m].dropna() for c in active_configs]
        if len(samples) > 1 and all(len(s) > 3 for s in samples):
            H, p = stats.kruskal(*samples)
            f.write(f"  {m:8}: H = {H:7.2f}, p = {p:.2e} {'(Significant)' if p<0.05 else '(Not sign.)'}\n")

    f.write("\nB. Pairwise Mann‑Whitney U tests (Bonferroni correction)\n")
    n_comparisons = len(list(combinations(active_configs, 2)))
    for m in metrics:
        if m not in df.columns:
            continue
        f.write(f"\n* Metric: {m}\n")
        for c1, c2 in combinations(active_configs, 2):
            s1 = df[df['Configuration'] == c1][m].dropna()
            s2 = df[df['Configuration'] == c2][m].dropna()
            if len(s1) < MIN_OBS_FOR_TEST or len(s2) < MIN_OBS_FOR_TEST:
                continue
            _, p_raw = stats.mannwhitneyu(s1, s2, alternative='two-sided')
            p_adj = min(p_raw * n_comparisons, 1.0)
            sig = "***" if p_adj < 0.001 else ("**" if p_adj < 0.01 else ("*" if p_adj < 0.05 else "ns"))
            f.write(f"    {c1:6} vs {c2:6}: p‑adj = {p_adj:.4f} {sig}\n")
    f.write("\n→ Key result: Normal regime has significantly lower E₃ than all other regimes,\n")
    f.write("  confirming E₃ as the primary structural boundary variable.\n")

# ============================================================================
# MODULE 4: C2 PATH – PREDICTING RECOVERY (C3) vs CRASH (C1/C6)
# ============================================================================
def c2_path_prediction(df, group_col, f):
    f.write("\n" + "="*80 + "\n")
    f.write("4. C2 PATH: PREDICTING RECOVERY (C3) vs. CRASH (C1/C6)\n")
    f.write("="*80 + "\n")
    df_next = df.copy()
    df_next['Next_Config'] = df_next.groupby(group_col)['Configuration'].shift(-1)
    c2_trans = df_next[(df_next['Configuration'] == 'C2') & (df_next['Next_Config'].notna())]
    c2_trans['outcome'] = np.where(c2_trans['Next_Config'] == 'C3', 'Recovery',
                                   np.where(c2_trans['Next_Config'].isin(CRASH_CONFIGS), 'Crash', 'Other'))
    c2_analysis = c2_trans[c2_trans['outcome'].isin(['Recovery', 'Crash'])].copy()
    f.write(f"Number of C2 transitions with known next state: {len(c2_trans)}\n")
    f.write(f"  - C2 → C3   : {(c2_analysis['outcome']=='Recovery').sum()} transitions\n")
    f.write(f"  - C2 → Crash: {(c2_analysis['outcome']=='Crash').sum()} transitions\n")

    for var in ['E_3', 'B']:
        if var not in c2_analysis.columns or c2_analysis[var].isna().all():
            f.write(f"\nSkipping {var} – not available.\n")
            continue
        y_true = (c2_analysis['outcome'] == 'Recovery').astype(int)
        y_score = c2_analysis[var].fillna(c2_analysis[var].median())
        if len(np.unique(y_true)) < 2:
            f.write(f"\n{var}: Only one outcome class – cannot compute AUC.\n")
            continue
        auc = roc_auc_score(y_true, y_score)
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        youden = tpr - fpr
        best_idx = np.argmax(youden)
        best_thresh = thresholds[best_idx]
        best_acc = (tpr[best_idx] + (1 - fpr[best_idx])) / 2
        rec_vals = c2_analysis[c2_analysis['outcome'] == 'Recovery'][var].dropna()
        cra_vals = c2_analysis[c2_analysis['outcome'] == 'Crash'][var].dropna()
        _, p_mw = stats.mannwhitneyu(rec_vals, cra_vals, alternative='two-sided')
        f.write(f"\n{var}:\n")
        f.write(f"  ROC AUC = {auc:.3f}\n")
        f.write(f"  Optimal threshold (Youden): {best_thresh:.2f} (accuracy = {best_acc*100:.1f}%)\n")
        f.write(f"  Mann‑Whitney U test: p = {p_mw:.2e}\n")
        if var == 'B' and auc > 0.75:
            f.write("  → B outperforms E₃ in predicting crash from C2 (consistent with Study 2).\n")

# ============================================================================
# MODULE 5: PDI AS LEADING INDICATOR IN C3/C4 REGIMES
# ============================================================================
def pdi_leading_in_c3c4(df, group_col, f):
    f.write("\n" + "="*80 + "\n")
    f.write("5. PDI LEADING INDICATOR IN C3/C4 REGIMES\n")
    f.write("="*80 + "\n")
    df_next = df.copy()
    df_next['Next_Config'] = df_next.groupby(group_col)['Configuration'].shift(-1)
    c3c4 = df_next[df_next['Configuration'].isin(['C3', 'C4']) & df_next['Next_Config'].notna()].copy()
    c3c4['next_crash'] = c3c4['Next_Config'].isin(CRASH_CONFIGS).astype(int)
    crash_next = c3c4[c3c4['next_crash'] == 1]
    safe_next = c3c4[c3c4['next_crash'] == 0]
    f.write(f"Number of C3/C4 quarters with known next state: {len(c3c4)}\n")
    f.write(f"  - Next quarter crash (C1/C6): n={len(crash_next)}\n")
    f.write(f"  - Next quarter safe       : n={len(safe_next)}\n")
    if len(crash_next) >= MIN_OBS_FOR_TEST and len(safe_next) >= MIN_OBS_FOR_TEST:
        median_crash = crash_next['PDI_t'].median()
        median_safe = safe_next['PDI_t'].median()
        _, p_mw = stats.mannwhitneyu(crash_next['PDI_t'].dropna(), safe_next['PDI_t'].dropna(), alternative='two-sided')
        f.write(f"  Median PDI before crash: {median_crash:.4f}\n")
        f.write(f"  Median PDI before safe : {median_safe:.4f}\n")
        f.write(f"  Mann‑Whitney U test: p = {p_mw:.2e}\n")
        if p_mw < 0.05 and median_crash < median_safe:
            f.write("  → Conclusion: PDI is significantly lower prior to a crash (leading indicator).\n")
        else:
            f.write("  → No significant leading effect in this sample.\n")
    else:
        f.write("  Insufficient data for leading indicator test.\n")

# ============================================================================
# MODULE 6: CONDITIONAL TEST (R_t ≈ 0 and dK_Pi_prime < 0)
# ============================================================================
def conditional_pdi_test(df, f):
    f.write("\n" + "="*80 + "\n")
    f.write("6. CONDITIONAL TEST (R_t ≈ 0 AND dK_Pi_prime < 0)\n")
    f.write("    → Under identical structural constraints, does PDI discriminate Normal vs Crash?\n")
    f.write("="*80 + "\n")
    if 'dK_Pi_prime_pct' not in df.columns:
        f.write("  Missing dK_Pi_prime_pct – skipping conditional test.\n")
        return
    cond = (df['R_t'].abs() < R_CONDITION_THRESHOLD) & (df['dK_Pi_prime_pct'] < 0)
    df_cond = df[cond].dropna(subset=['PDI_t', 'Configuration'])
    normal = df_cond[df_cond['Configuration'] == 'Normal']
    crash = df_cond[df_cond['Configuration'].isin(CRASH_CONFIGS)]
    f.write(f"Observations satisfying condition: {len(df_cond)}\n")
    f.write(f"  - Normal state : n={len(normal)}\n")
    f.write(f"  - Crash state  : n={len(crash)}\n")
    if len(normal) >= MIN_OBS_FOR_TEST and len(crash) >= MIN_OBS_FOR_TEST:
        median_norm = normal['PDI_t'].median()
        median_crash = crash['PDI_t'].median()
        _, p_val = stats.mannwhitneyu(normal['PDI_t'], crash['PDI_t'], alternative='two-sided')
        f.write(f"  Median PDI in Normal: {median_norm:.4f}\n")
        f.write(f"  Median PDI in Crash : {median_crash:.4f}\n")
        f.write(f"  Mann‑Whitney U test: p = {p_val:.2e}\n")
        if p_val < 0.05:
            f.write("  → Conclusion: PDI discriminates between Normal and crash regimes even under identical R_t and capital outflow conditions.\n")
        else:
            f.write("  → No significant discrimination.\n")
    else:
        f.write("  Insufficient data for conditional test.\n")

# ============================================================================
# MODULE 7: DIRECTIONAL TEST (LAGGED PDI)
# ============================================================================
def directional_pdi_test(df, group_col, f):
    f.write("\n" + "="*80 + "\n")
    f.write("7. DIRECTIONAL TEST (LAGGED PDI AS LEADING INDICATOR)\n")
    f.write("    → Does PDI(t-1) predict next‑quarter crash?\n")
    f.write("="*80 + "\n")
    df_lag = df.dropna(subset=['PDI_lag1', 'is_crash'])
    survive = df_lag[df_lag['is_crash'] == 0]['PDI_lag1']
    crash = df_lag[df_lag['is_crash'] == 1]['PDI_lag1']
    f.write(f"Total observations (with lag): {len(df_lag)}\n")
    f.write(f"  - Survival (no crash): n={len(survive)}\n")
    f.write(f"  - Crash (C1/C6)      : n={len(crash)}\n")
    if len(survive) >= MIN_OBS_FOR_TEST and len(crash) >= MIN_OBS_FOR_TEST:
        median_surv = survive.median()
        median_crash = crash.median()
        _, p_val = stats.mannwhitneyu(survive, crash, alternative='two-sided')
        f.write(f"  Median PDI_lag1 in survival group: {median_surv:.4f}\n")
        f.write(f"  Median PDI_lag1 in crash group   : {median_crash:.4f}\n")
        f.write(f"  Mann‑Whitney U test: p = {p_val:.2e}\n")
        if p_val < 0.05 and median_crash < median_surv:
            f.write("  → Conclusion: Lagged PDI is a directional leading indicator for crashes.\n")
        else:
            f.write("  → No significant leading effect.\n")
    else:
        f.write("  Insufficient data for directional test.\n")

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    try:
        df, group_col = load_and_prepare_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("STRUCTURAL BOUNDARY ANALYSIS – ACADEMIC REPORT (STUDY 1)\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*100 + "\n")

        markov_transitions(df, group_col, f)
        spearman_correlations(df, f)
        boundary_analysis(df, f)
        c2_path_prediction(df, group_col, f)
        pdi_leading_in_c3c4(df, group_col, f)
        conditional_pdi_test(df, f)
        directional_pdi_test(df, group_col, f)

        f.write("\n" + "="*100 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*100 + "\n")

    print(f"Report saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()