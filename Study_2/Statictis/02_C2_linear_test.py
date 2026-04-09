"""
02_DynamicsLeading.py

Academic Title:
    Leading Indicators and Dynamic Path Analysis for Crisis Prediction

Objectives:
    This script investigates the predictive power of several variables
    for transitions into crash states (C1/C6) versus non‑crash states,
    both globally and at the sector level.

Methods Included:
    1. C2 Path Analysis:
       - Compare E_3 and B = E_3 - (1 + PGR_t) in predicting whether a C2 state
         is followed by C3 (recovery) or by a crash (C1/C6).
       - Compute ROC curves, AUC, optimal threshold (Youden index), and accuracy.
       - Mann‑Whitney U test for differences between the two groups.

    2. PDI Leading Indicator in C3/C4:
       - For quarters in C3 or C4, test whether current PDI_t is lower
         when the next quarter is a crash (C1/C6) versus non‑crash.
       - Mann‑Whitney U test.

    3. Conditional Test (Structural Constraint):
       - Under the condition R_t ≈ 0 (discharging capital) and dK_Pi_prime < 0
         (capital outflow), compare PDI_t between Normal and crash states (C1/C6).

    4. Directional Test (Lagged PDI):
       - Test whether PDI lagged by one quarter (PDI_lag1) differs between
         periods that end in a crash vs. those that do not.

Output Files (saved in data/results/):
    - dynamics_leading_report.txt          : Full academic report.
    - dynamics_c2_roc.csv                  : ROC data for E_3 and B (thresholds, TPR, FPR).
    - dynamics_pdi_leading.csv             : PDI leading indicator results by scope.
    - dynamics_conditional.csv             : Conditional test results by scope.
    - dynamics_directional.csv             : Directional (lagged) test results by scope.

Dependencies:
    pandas, numpy, scipy, scikit‑learn, pathlib
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_FILE = Path('../data/final_panel.csv')
OUTPUT_DIR = Path('results')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TXT_REPORT = OUTPUT_DIR / 'report/02_dynamics_leading_report.txt'
CSV_C2_ROC = OUTPUT_DIR / 'table/02_dynamics_c2_roc.csv'
CSV_PDI_LEADING = OUTPUT_DIR / 'table/02_dynamics_pdi_leading.csv'
CSV_CONDITIONAL = OUTPUT_DIR / 'table/02_dynamics_conditional.csv'
CSV_DIRECTIONAL = OUTPUT_DIR / 'table/02_dynamics_directional.csv'

VALID_STATES = ['Normal', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
CRASH_STATES = ['C1', 'C6']
SAFE_STATES = ['C3', 'C4', 'C5', 'Normal']  # for leading indicator test
RANDOM_SEED = 42
MIN_OBS_FOR_TEST = 3
MIN_OBS_DIRECTIONAL = 5
R_CONDITION_THRESHOLD = 0.05

np.random.seed(RANDOM_SEED)


# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================

def load_and_prepare_data():
    """
    Load the panel dataset, apply the formal gate (Regime_Label overrides
    Configuration to 'Normal'), filter to valid states, and compute derived
    variables if missing.
    """
    df = pd.read_csv(DATA_FILE)
    df['period_end'] = pd.to_datetime(df['period_end'])
    
    if 'Regime_Label' in df.columns:
        df['Configuration'] = np.where(
            df['Regime_Label'] == 'Normal_Regime',
            'Normal',
            df['Configuration']
        )
    
    df = df[df['Configuration'].isin(VALID_STATES)].copy()
    
    numeric_cols = ['E_3', 'R_t', 'PDI_t', 'PGR_t', 'K_Pi_prime', 'dK_Pi_prime']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Sort for proper lag/lead operations
    df = df.sort_values(['Ticker', 'period_end']).reset_index(drop=True)
    
    # Compute PGR_t if missing (using V_Prod_base growth)
    if 'PGR_t' not in df.columns and 'V_Prod_base' in df.columns:
        group_col = 'Cycle_ID' if 'Cycle_ID' in df.columns else 'Ticker'
        df['PGR_t'] = df.groupby(group_col)['V_Prod_base'].pct_change()
    
    # Compute B = E_3 - (1 + PGR_t) if PGR_t is available
    if 'PGR_t' in df.columns:
        df['B'] = df['E_3'] - (1 + df['PGR_t'])
    
    print(f"Data loaded: {len(df):,} observations.")
    return df


def get_group_column(df):
    """Return the appropriate column for grouping (Cycle_ID preferred)."""
    return 'Cycle_ID' if 'Cycle_ID' in df.columns else 'Ticker'


# ============================================================================
# C2 PATH ANALYSIS (E_3 vs B)
# ============================================================================

def c2_path_analysis(df, group_col):
    """
    For transitions from C2 to either C3 (recovery) or crash (C1/C6),
    evaluate the predictive performance of E_3 and B.

    Returns:
        results (dict) : Contains AUC, optimal threshold, accuracy, p-values.
        roc_df (DataFrame): ROC curve points (fpr, tpr, threshold) for both variables.
        transition_data (DataFrame): Raw data used for analysis.
    """
    if 'B' not in df.columns:
        print("Warning: 'B' not available. Skipping C2 path analysis.")
        return None, None, None
    
    # Create next state variable
    df_sorted = df.sort_values([group_col, 'period_end']).copy()
    df_sorted['next_state'] = df_sorted.groupby(group_col)['Configuration'].shift(-1)
    
    # Filter C2 rows with known next state (C3, C1, or C6)
    c2_trans = df_sorted[df_sorted['Configuration'] == 'C2'].dropna(subset=['next_state'])
    c2_trans = c2_trans[c2_trans['next_state'].isin(['C3', 'C1', 'C6'])]
    c2_trans = c2_trans.dropna(subset=['E_3', 'B'])
    
    if len(c2_trans) == 0:
        print("Insufficient C2 transitions for analysis.")
        return None, None, None
    
    # Binary target: 1 = crash (C1/C6), 0 = recovery (C3)
    y = (c2_trans['next_state'].isin(CRASH_STATES)).astype(int)
    
    # --- E_3 ---
    fpr_e3, tpr_e3, th_e3 = roc_curve(y, c2_trans['E_3'].values)
    auc_e3 = auc(fpr_e3, tpr_e3)
    youden_idx = np.argmax(tpr_e3 - fpr_e3)
    best_th_e3 = th_e3[youden_idx]
    acc_e3 = (tpr_e3[youden_idx] + 1 - fpr_e3[youden_idx]) / 2
    
    # --- B ---
    fpr_b, tpr_b, th_b = roc_curve(y, c2_trans['B'].values)
    auc_b = auc(fpr_b, tpr_b)
    youden_idx_b = np.argmax(tpr_b - fpr_b)
    best_th_b = th_b[youden_idx_b]
    acc_b = (tpr_b[youden_idx_b] + 1 - fpr_b[youden_idx_b]) / 2
    
    # Mann‑Whitney U tests
    e3_c3 = c2_trans[c2_trans['next_state'] == 'C3']['E_3'].dropna()
    e3_crash = c2_trans[c2_trans['next_state'].isin(CRASH_STATES)]['E_3'].dropna()
    p_e3 = stats.mannwhitneyu(e3_c3, e3_crash, alternative='two-sided')[1] if len(e3_c3) >= MIN_OBS_FOR_TEST and len(e3_crash) >= MIN_OBS_FOR_TEST else np.nan
    
    b_c3 = c2_trans[c2_trans['next_state'] == 'C3']['B'].dropna()
    b_crash = c2_trans[c2_trans['next_state'].isin(CRASH_STATES)]['B'].dropna()
    p_b = stats.mannwhitneyu(b_c3, b_crash, alternative='two-sided')[1] if len(b_c3) >= MIN_OBS_FOR_TEST and len(b_crash) >= MIN_OBS_FOR_TEST else np.nan
    
    results = {
        'n_sequences': len(c2_trans),
        'n_to_c3': len(e3_c3),
        'n_to_crash': len(e3_crash),
        'auc_e3': auc_e3,
        'auc_b': auc_b,
        'best_th_e3': best_th_e3,
        'best_th_b': best_th_b,
        'acc_e3': acc_e3,
        'acc_b': acc_b,
        'p_e3': p_e3,
        'p_b': p_b
    }
    
    roc_df = pd.DataFrame({
        'variable': ['E_3'] * len(fpr_e3) + ['B'] * len(fpr_b),
        'fpr': np.concatenate([fpr_e3, fpr_b]),
        'tpr': np.concatenate([tpr_e3, tpr_b]),
        'threshold': np.concatenate([th_e3, th_b])
    })
    
    return results, roc_df, c2_trans


# ============================================================================
# PDI LEADING INDICATOR IN C3/C4
# ============================================================================

def pdi_leading_indicator(df, group_col):
    """
    For observations in C3 or C4, test whether current PDI_t is lower
    when the next quarter is a crash (C1/C6) than when it is safe.
    Returns a dictionary with sample sizes, medians, and p-value.
    """
    df = df.copy()
    df['is_crash'] = df['Configuration'].isin(CRASH_STATES).astype(int)
    df['crash_next'] = df.groupby(group_col)['is_crash'].shift(-1)
    
    c3c4 = df[df['Configuration'].isin(['C3', 'C4'])].dropna(subset=['PDI_t', 'crash_next'])
    if len(c3c4) == 0:
        return None
    
    crash_next_pdi = c3c4[c3c4['crash_next'] == 1]['PDI_t']
    safe_next_pdi = c3c4[c3c4['crash_next'] == 0]['PDI_t']
    
    if len(crash_next_pdi) >= MIN_OBS_FOR_TEST and len(safe_next_pdi) >= MIN_OBS_FOR_TEST:
        _, p_val = stats.mannwhitneyu(crash_next_pdi, safe_next_pdi, alternative='two-sided')
    else:
        p_val = np.nan
    
    return {
        'n_c3c4': len(c3c4),
        'n_crash_next': len(crash_next_pdi),
        'n_safe_next': len(safe_next_pdi),
        'median_pdi_crash_next': crash_next_pdi.median() if len(crash_next_pdi) > 0 else np.nan,
        'median_pdi_safe_next': safe_next_pdi.median() if len(safe_next_pdi) > 0 else np.nan,
        'p_value': p_val
    }


# ============================================================================
# CONDITIONAL TEST (R_t ≈ 0 AND dK < 0)
# ============================================================================

def conditional_test(df):
    """
    Under the condition R_t < threshold and dK_Pi_prime < 0 (if available),
    compare PDI_t between Normal and crash states (C1/C6).
    """
    condition = (df['R_t'] < R_CONDITION_THRESHOLD)
    if 'dK_Pi_prime' in df.columns:
        condition = condition & (df['dK_Pi_prime'] < 0)
    
    df_cond = df[condition].dropna(subset=['PDI_t', 'Configuration'])
    normal_pdi = df_cond[df_cond['Configuration'] == 'Normal']['PDI_t']
    crash_pdi = df_cond[df_cond['Configuration'].isin(CRASH_STATES)]['PDI_t']
    
    if len(normal_pdi) >= MIN_OBS_FOR_TEST and len(crash_pdi) >= MIN_OBS_FOR_TEST:
        _, p_val = stats.mannwhitneyu(normal_pdi, crash_pdi, alternative='two-sided')
    else:
        p_val = np.nan
    
    return {
        'n_condition': len(df_cond),
        'n_normal': len(normal_pdi),
        'n_crash': len(crash_pdi),
        'median_pdi_normal': normal_pdi.median() if len(normal_pdi) > 0 else np.nan,
        'median_pdi_crash': crash_pdi.median() if len(crash_pdi) > 0 else np.nan,
        'p_value': p_val
    }


# ============================================================================
# DIRECTIONAL TEST (LAGGED PDI)
# ============================================================================

def directional_test(df, group_col):
    """
    Test whether PDI lagged by one quarter (PDI_lag1) differs between
    periods that end in a crash vs. those that do not.
    """
    df = df.copy()
    df['is_crash'] = df['Configuration'].isin(CRASH_STATES).astype(int)
    df['PDI_lag1'] = df.groupby(group_col)['PDI_t'].shift(1)
    df_dir = df.dropna(subset=['is_crash', 'PDI_lag1'])
    
    survive_pdi = df_dir[df_dir['is_crash'] == 0]['PDI_lag1']
    crash_pdi = df_dir[df_dir['is_crash'] == 1]['PDI_lag1']
    
    if len(survive_pdi) >= MIN_OBS_DIRECTIONAL and len(crash_pdi) >= MIN_OBS_DIRECTIONAL:
        _, p_val = stats.mannwhitneyu(survive_pdi, crash_pdi, alternative='two-sided')
    else:
        p_val = np.nan
    
    return {
        'n_obs': len(df_dir),
        'n_survive': len(survive_pdi),
        'n_crash': len(crash_pdi),
        'median_pdi_lag_survive': survive_pdi.median() if len(survive_pdi) > 0 else np.nan,
        'median_pdi_lag_crash': crash_pdi.median() if len(crash_pdi) > 0 else np.nan,
        'p_value': p_val
    }


# ============================================================================
# ACADEMIC REPORT GENERATION
# ============================================================================

def write_academic_report(global_results, sector_results, f):
    """
    Write the full academic report to the output file handle.
    """
    f.write("=" * 100 + "\n")
    f.write("DYNAMICS AND LEADING INDICATORS – ACADEMIC REPORT\n")
    f.write("Global Market and Sector-Level Analysis\n")
    f.write("=" * 100 + "\n\n")
    
    # ----- 1. Global -----
    f.write("1. GLOBAL MARKET\n")
    f.write("-" * 50 + "\n")
    
    # C2 Path
    if global_results.get('c2'):
        r = global_results['c2']
        f.write("\n[1.1] C2 PATH: PREDICTING RECOVERY (C3) vs. CRASH (C1/C6)\n")
        f.write(f"      Number of C2 transitions with known next state: {r['n_sequences']}\n")
        f.write(f"      - C2 → C3   : {r['n_to_c3']} transitions\n")
        f.write(f"      - C2 → Crash: {r['n_to_crash']} transitions\n")
        f.write(f"      ROC AUC:\n")
        f.write(f"          E_3 : {r['auc_e3']:.3f}\n")
        f.write(f"          B   : {r['auc_b']:.3f}\n")
        f.write(f"      Optimal threshold (Youden index):\n")
        f.write(f"          E_3 : {r['best_th_e3']:.2f} (accuracy = {r['acc_e3']:.1%})\n")
        f.write(f"          B   : {r['best_th_b']:.2f} (accuracy = {r['acc_b']:.1%})\n")
        if not np.isnan(r['p_e3']):
            f.write(f"      Mann‑Whitney U test (E_3): p = {r['p_e3']:.2e}\n")
        if not np.isnan(r['p_b']):
            f.write(f"      Mann‑Whitney U test (B):  p = {r['p_b']:.2e}\n")
        if r['auc_b'] > r['auc_e3']:
            f.write("      → Conclusion: B outperforms E_3 in predicting crash from C2.\n")
        else:
            f.write("      → Conclusion: E_3 is at least as good as B.\n")
    
    # PDI leading indicator
    if global_results.get('pdi_leading'):
        r = global_results['pdi_leading']
        f.write("\n[1.2] PDI LEADING INDICATOR IN C3/C4\n")
        f.write(f"      Number of C3/C4 quarters with known next state: {r['n_c3c4']}\n")
        f.write(f"      - Next quarter crash (C1/C6): n={r['n_crash_next']}, median PDI = {r['median_pdi_crash_next']:.4f}\n")
        f.write(f"      - Next quarter safe       : n={r['n_safe_next']}, median PDI = {r['median_pdi_safe_next']:.4f}\n")
        if not np.isnan(r['p_value']):
            f.write(f"      Mann‑Whitney U test: p = {r['p_value']:.2e}\n")
            if r['p_value'] < 0.05:
                f.write("      → Conclusion: PDI is significantly lower prior to a crash (leading indicator).\n")
            else:
                f.write("      → Conclusion: No sufficient evidence of leading indicator property.\n")
    
    # Conditional test
    if global_results.get('conditional'):
        r = global_results['conditional']
        f.write("\n[1.3] CONDITIONAL TEST (R_t ≈ 0 and dK_Pi_prime < 0)\n")
        f.write(f"      Observations satisfying condition: {r['n_condition']}\n")
        f.write(f"      - Normal state: n={r['n_normal']}, median PDI = {r['median_pdi_normal']:.4f}\n")
        f.write(f"      - Crash state (C1/C6): n={r['n_crash']}, median PDI = {r['median_pdi_crash']:.4f}\n")
        if not np.isnan(r['p_value']):
            f.write(f"      Mann‑Whitney U test: p = {r['p_value']:.2e}\n")
            if r['p_value'] < 0.05:
                f.write("      → Conclusion: Under identical structural constraints, PDI discriminates between Normal and crash regimes.\n")
    
    # Directional test
    if global_results.get('directional'):
        r = global_results['directional']
        f.write("\n[1.4] DIRECTIONAL TEST (LAGGED PDI)\n")
        f.write(f"      Total observations (with lag): {r['n_obs']}\n")
        f.write(f"      - Survival (no crash): n={r['n_survive']}, median PDI_lag1 = {r['median_pdi_lag_survive']:.4f}\n")
        f.write(f"      - Crash (C1/C6)      : n={r['n_crash']}, median PDI_lag1 = {r['median_pdi_lag_crash']:.4f}\n")
        if not np.isnan(r['p_value']):
            f.write(f"      Mann‑Whitney U test: p = {r['p_value']:.2e}\n")
            if r['p_value'] < 0.05:
                f.write("      → Conclusion: Lagged PDI is a directional leading indicator for crashes.\n")
    
    # ----- 2. Sector summaries -----
    if sector_results:
        f.write("\n\n2. SECTOR‑LEVEL SUMMARIES (≥50 observations)\n")
        f.write("-" * 50 + "\n")
        for sec, res in sector_results.items():
            f.write(f"\n--- {sec} ---\n")
            if res.get('c2'):
                r = res['c2']
                f.write(f"   C2 path: AUC(E_3)={r['auc_e3']:.3f}, AUC(B)={r['auc_b']:.3f}\n")
            if res.get('pdi_leading'):
                r = res['pdi_leading']
                if not np.isnan(r['p_value']):
                    f.write(f"   PDI leading (C3/C4): p = {r['p_value']:.2e}\n")
            if res.get('directional'):
                r = res['directional']
                if not np.isnan(r['p_value']):
                    f.write(f"   Directional (lagged PDI): p = {r['p_value']:.2e}\n")
    
    f.write("\n" + "=" * 100 + "\n")
    f.write("Detailed ROC data and full results are available in the accompanying CSV files.\n")
    f.write("=" * 100 + "\n")


# ============================================================================
# CSV EXPORT FUNCTIONS
# ============================================================================

def export_csv_files(global_results, sector_results, roc_global, roc_sectors):
    """
    Export all DataFrames to CSV files.
    """
    # 1. C2 ROC data
    all_roc = []
    if roc_global is not None:
        roc_global['scope'] = 'Global'
        all_roc.append(roc_global)
    for sec, roc_df in roc_sectors.items():
        if roc_df is not None:
            roc_df['scope'] = f'Sector_{sec}'
            all_roc.append(roc_df)
    if all_roc:
        pd.concat(all_roc, ignore_index=True).to_csv(CSV_C2_ROC, index=False, encoding='utf-8-sig')
    
    # 2. PDI leading results
    pdi_rows = []
    if global_results.get('pdi_leading'):
        pdi_rows.append({'scope': 'Global', **global_results['pdi_leading']})
    for sec, res in sector_results.items():
        if res.get('pdi_leading'):
            pdi_rows.append({'scope': f'Sector_{sec}', **res['pdi_leading']})
    if pdi_rows:
        pd.DataFrame(pdi_rows).to_csv(CSV_PDI_LEADING, index=False, encoding='utf-8-sig')
    
    # 3. Conditional test results
    cond_rows = []
    if global_results.get('conditional'):
        cond_rows.append({'scope': 'Global', **global_results['conditional']})
    for sec, res in sector_results.items():
        if res.get('conditional'):
            cond_rows.append({'scope': f'Sector_{sec}', **res['conditional']})
    if cond_rows:
        pd.DataFrame(cond_rows).to_csv(CSV_CONDITIONAL, index=False, encoding='utf-8-sig')
    
    # 4. Directional test results
    dir_rows = []
    if global_results.get('directional'):
        dir_rows.append({'scope': 'Global', **global_results['directional']})
    for sec, res in sector_results.items():
        if res.get('directional'):
            dir_rows.append({'scope': f'Sector_{sec}', **res['directional']})
    if dir_rows:
        pd.DataFrame(dir_rows).to_csv(CSV_DIRECTIONAL, index=False, encoding='utf-8-sig')
    
    print(f"All CSV files exported to {OUTPUT_DIR}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 80)
    print("DYNAMICS AND LEADING INDICATORS – ACADEMIC ANALYSIS")
    print("C2 Path, PDI Leading Indicator, Conditional and Directional Tests")
    print("=" * 80)
    
    # Load data
    df = load_and_prepare_data()
    group_col = get_group_column(df)
    
    # Global analysis
    print("\nRunning global analysis...")
    c2_res_global, roc_global, _ = c2_path_analysis(df, group_col)
    pdi_res_global = pdi_leading_indicator(df, group_col)
    cond_res_global = conditional_test(df)
    dir_res_global = directional_test(df, group_col)
    
    global_results = {
        'c2': c2_res_global,
        'pdi_leading': pdi_res_global,
        'conditional': cond_res_global,
        'directional': dir_res_global
    }
    
    # Sector analysis (only sectors with at least 50 observations)
    sector_results = {}
    roc_sectors = {}
    if 'Sector' in df.columns:
        sectors = df['Sector'].dropna().unique()
        for sec in sectors:
            df_sec = df[df['Sector'] == sec].copy()
            if len(df_sec) < 50:
                continue
            print(f"Processing sector: {sec}")
            c2_res, roc_sec, _ = c2_path_analysis(df_sec, group_col)
            pdi_res = pdi_leading_indicator(df_sec, group_col)
            cond_res = conditional_test(df_sec)
            dir_res = directional_test(df_sec, group_col)
            sector_results[sec] = {
                'c2': c2_res,
                'pdi_leading': pdi_res,
                'conditional': cond_res,
                'directional': dir_res
            }
            if roc_sec is not None:
                roc_sectors[sec] = roc_sec
    
    # Write academic report
    with open(TXT_REPORT, 'w', encoding='utf-8') as f:
        write_academic_report(global_results, sector_results, f)
    
    # Export CSVs
    export_csv_files(global_results, sector_results, roc_global, roc_sectors)
    
    print(f"\n✅ Analysis completed successfully.")
    print(f"   Academic report: {TXT_REPORT}")
    print(f"   All CSV outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()