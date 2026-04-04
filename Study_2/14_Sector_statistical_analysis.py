"""
STEP 13: COMPREHENSIVE STATISTICAL & ROBUSTNESS ANALYSIS (GLOBAL & SECTOR)
Consolidates all empirical tests for the research paper:
1. Markov Transition Matrix
2. Descriptive Statistics (PDI & Delta PDI focus)
3. Spearman Rank Correlation (Speculative Paradox)
4. Kruskal-Wallis & Post-Hoc Tests (Boundary Analysis)
5. Conditional & Directional Dynamics Tests
6. Robustness Tests (Temporal Split & Bootstrapping)

Output: 
- Console log
- data/results/comprehensive_report.txt
- data/results/markov_transitions.csv
- data/results/comprehensive_stats.csv
"""

import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
DATA_FILE = Path('data/final_panel.csv')
RESULTS_DIR = Path('data/results')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

REPORT_FILE = RESULTS_DIR / 'comprehensive_report.txt'
CSV_STATS_FILE = RESULTS_DIR / 'comprehensive_stats.csv'
CSV_MARKOV_FILE = RESULTS_DIR / 'markov_transitions.csv'

VALID_CONFIGS = ['Normal', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']

def cprint(text="", f=None):
    """Prints to console and writes to file if file handle is provided."""
    print(text)
    if f is not None:
        f.write(str(text) + "\n")

def load_and_prep_data():
    """Loads panel data and calculates required lagged/delta variables."""
    print("🔄 Loading and preparing panel data...")
    try:
        df = pd.read_csv(DATA_FILE)
        df['period_end'] = pd.to_datetime(df['period_end'])
        
        # Ensure Configuration is clean (Apply Formal Gate override explicitly if missed)
        if 'Regime_Label' in df.columns:
            df['Configuration'] = np.where(df['Regime_Label'] == 'Normal_Regime', 'Normal', df['Configuration'])
            
        # Coerce numeric types
        numeric_cols = ['E_3', 'R_t', 'PDI_t', 'dK_Pi_prime_pct', 'PGR_t']
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
                df[c] = df[c].replace([np.inf, -np.inf], np.nan)
                
        # Sort values by Ticker and Time to ensure momentum dynamics are perfectly aligned
        df = df.sort_values(['Ticker', 'period_end']).reset_index(drop=True)
        
        if 'PDI_t' in df.columns:
            df['d_PDI_t'] = df.groupby('Ticker')['PDI_t'].diff()
            df['PDI_lag1'] = df.groupby('Ticker')['PDI_t'].shift(1)
        
        df['is_Crash_Regime'] = np.where(df['Configuration'].isin(['C1', 'C6']), 1, 0)
        
        print(f"✅ Data loaded: {len(df):,} observations across {df['Ticker'].nunique()} companies.")
        return df
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return None

def analyze_markov_transitions(df, group_title, f=None, markov_list=None):
    cprint("\n" + "="*80, f)
    cprint("1. MARKOV TRANSITION MATRIX (FULL LIFECYCLE)", f)
    cprint("="*80, f)
    
    df_sorted = df.copy()
    df_sorted['Next_Config'] = df_sorted.groupby('Ticker')['Configuration'].shift(-1)
    
    transitions = df_sorted.dropna(subset=['Next_Config'])
    trans_data = transitions[
        (transitions['Configuration'].isin(VALID_CONFIGS)) &
        (transitions['Next_Config'].isin(VALID_CONFIGS))
    ]
    
    if len(trans_data) == 0:
        cprint("  > Not enough valid data for Markov transitions.", f)
        return

    matrix = pd.crosstab(
        trans_data['Configuration'],
        trans_data['Next_Config'],
        normalize='index'
    ) * 100
    
    matrix = matrix.reindex(index=VALID_CONFIGS, columns=VALID_CONFIGS, fill_value=0)
    cprint(matrix.round(1).to_string(), f)

    # Append to the export list
    if markov_list is not None:
        for c_state in VALID_CONFIGS:
            for n_state in VALID_CONFIGS:
                prob = matrix.loc[c_state, n_state] if c_state in matrix.index and n_state in matrix.columns else 0.0
                markov_list.append({
                    'Scope': group_title,
                    'Current_State': c_state,
                    'Next_State': n_state,
                    'Transition_Prob_Pct': round(prob, 2)
                })

def analyze_descriptive_pdi(df, group_title, f=None, stats_list=None):
    cprint("\n" + "="*80, f)
    cprint("2. DESCRIPTIVE STATISTICS (PDI_t & ΔPDI_t)", f)
    cprint("="*80, f)
    
    if 'PDI_t' not in df.columns or 'd_PDI_t' not in df.columns:
        cprint("  > Missing PDI columns.", f)
        return
        
    summary = df[df['Configuration'].isin(VALID_CONFIGS)].groupby('Configuration')[['PDI_t', 'd_PDI_t']].agg(['mean', 'median'])
    cprint(summary.round(4).to_string(), f)
    
    if stats_list is not None:
        for config in summary.index:
            stats_list.append({'Scope': group_title, 'Test_Name': 'Descriptive', 'Target': 'PDI_t', 'Comparison': config, 'Stat_Type': 'Mean', 'Stat_Value': summary.loc[config, ('PDI_t', 'mean')], 'P_Value': np.nan, 'Interpretation': 'N/A'})
            stats_list.append({'Scope': group_title, 'Test_Name': 'Descriptive', 'Target': 'd_PDI_t', 'Comparison': config, 'Stat_Type': 'Mean', 'Stat_Value': summary.loc[config, ('d_PDI_t', 'mean')], 'P_Value': np.nan, 'Interpretation': 'N/A'})

def analyze_correlations(df, group_title, f=None, stats_list=None):
    cprint("\n" + "="*80, f)
    cprint("3. SPEARMAN RANK CORRELATION (SPECULATIVE PARADOX)", f)
    cprint("="*80, f)
    
    req_cols = ['E_3', 'R_t']
    if 'PDI_t' in df.columns: req_cols.extend(['PDI_t', 'd_PDI_t'])
        
    df_corr = df.dropna(subset=[c for c in req_cols if c in df.columns]).copy()
    if len(df_corr) < 5: 
        cprint("  > Not enough valid data.", f)
        return

    def log_corr(target, col1, col2):
        rho, p = stats.spearmanr(df_corr[col1], df_corr[col2], nan_policy='omit')
        sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))
        cprint(f"  {col1} vs {col2:7}: rho = {rho:6.4f} | p-value = {p:.2e} {sig}", f)
        if stats_list is not None:
            stats_list.append({'Scope': group_title, 'Test_Name': 'Spearman Correlation', 'Target': target, 'Comparison': 'Global', 'Stat_Type': 'Rho', 'Stat_Value': rho, 'P_Value': p, 'Interpretation': sig if sig else 'Not Sig'})

    log_corr('E_3 vs R_t', 'E_3', 'R_t')
    if 'PDI_t' in df.columns:
        log_corr('E_3 vs PDI_t', 'E_3', 'PDI_t')
        log_corr('E_3 vs d_PDI_t', 'E_3', 'd_PDI_t')

def analyze_kruskal_and_posthoc(df, group_title, f=None, stats_list=None):
    cprint("\n" + "="*80, f)
    cprint("4. STRUCTURAL BOUNDARY ANALYSIS (KRUSKAL-WALLIS & MANN-WHITNEY)", f)
    cprint("="*80, f)
    
    metrics = ['E_3', 'PGR_t', 'R_t', 'PDI_t', 'd_PDI_t', 'dK_Pi_prime_pct']
    valid_data_dict = {}
    
    cprint("A. Kruskal-Wallis H-Test (Global Variance)", f)
    for m in metrics:
        if m not in df.columns: continue
        samples = []
        configs = []
        for c in VALID_CONFIGS:
            s = df[df['Configuration'] == c][m].dropna()
            if len(s) > 3: 
                samples.append(s)
                configs.append(c)
                
        valid_data_dict[m] = {c: s for c, s in zip(configs, samples)}
        if len(samples) > 1:
            stat, p = stats.kruskal(*samples)
            sig = "Significant" if p < 0.05 else "Not Significant"
            cprint(f"  > {m:15}: H-stat = {stat:6.2f} | p-value = {p:.4e} ({sig})", f)
            if stats_list is not None:
                stats_list.append({'Scope': group_title, 'Test_Name': 'Kruskal-Wallis', 'Target': m, 'Comparison': 'All Valid Configs', 'Stat_Type': 'H-Stat', 'Stat_Value': stat, 'P_Value': p, 'Interpretation': sig})
    
    cprint("\nB. Post-Hoc Mann-Whitney U Test (Bonferroni Corrected)", f)
    num_comps = len(list(combinations(VALID_CONFIGS, 2)))
    
    for metric in ['E_3', 'PDI_t', 'd_PDI_t']:
        if metric in valid_data_dict:
            cprint(f"\n--- Metric: {metric} ---", f)
            available_configs = list(valid_data_dict[metric].keys())
            for c1, c2 in combinations(VALID_CONFIGS, 2):
                if c1 in available_configs and c2 in available_configs:
                    s1, s2 = valid_data_dict[metric][c1], valid_data_dict[metric][c2]
                    try:
                        _, p = stats.mannwhitneyu(s1, s2, alternative='two-sided')
                        p_adj = min(p * num_comps, 1.0)          
                        highlight = "📌" if "Normal" in [c1, c2] else "  "
                        sig = "Different" if p_adj < 0.05 else "Similar"
                        cprint(f" {highlight} [{c1:6} vs {c2:6}]: p-adj = {p_adj:.4f} ({sig})", f)
                        if stats_list is not None:
                            stats_list.append({'Scope': group_title, 'Test_Name': 'Mann-Whitney U', 'Target': metric, 'Comparison': f'{c1} vs {c2}', 'Stat_Type': 'P-adj', 'Stat_Value': np.nan, 'P_Value': p_adj, 'Interpretation': sig})
                    except ValueError: pass

def test_pdi_dynamics(df, group_title, f=None, stats_list=None):
    if 'PDI_t' not in df.columns or 'd_PDI_t' not in df.columns:
        return
        
    cprint("\n" + "="*80, f)
    cprint("5. DYNAMICS: CONDITIONAL AND LEADING INDICATOR TESTS", f)
    cprint("="*80, f)
    
    cprint("A. Conditional Test (Fixed Structure: R_t ≈ 0, discharging capital)", f)
    cond_mask = (df['R_t'] < 0.05) & (df['dK_Pi_prime_pct'] < 0)
    df_cond = df[cond_mask].dropna(subset=['PDI_t', 'd_PDI_t'])
    
    g_normal = df_cond[df_cond['Configuration'] == 'Normal']
    g_crash = df_cond[df_cond['Configuration'].isin(['C1', 'C6'])]
    
    if len(g_normal) >= 3 and len(g_crash) >= 3:
        _, p_pdi = stats.mannwhitneyu(g_normal['PDI_t'], g_crash['PDI_t'], alternative='two-sided')
        _, p_dpdi = stats.mannwhitneyu(g_normal['d_PDI_t'], g_crash['d_PDI_t'], alternative='two-sided')
        cprint(f"  > Normal (n={len(g_normal)}) vs C1/C6 (n={len(g_crash)})", f)
        cprint(f"  > Discrimination via PDI_t   : p-value = {p_pdi:.4e}", f)
        cprint(f"  > Discrimination via ΔPDI_t  : p-value = {p_dpdi:.4e}", f)
        
        if stats_list is not None:
            stats_list.append({'Scope': group_title, 'Test_Name': 'Conditional Dynamics', 'Target': 'PDI_t', 'Comparison': 'Normal vs C1/C6', 'Stat_Type': 'U-Test', 'Stat_Value': np.nan, 'P_Value': p_pdi, 'Interpretation': 'Significant' if p_pdi < 0.05 else 'Not Sig'})
            stats_list.append({'Scope': group_title, 'Test_Name': 'Conditional Dynamics', 'Target': 'd_PDI_t', 'Comparison': 'Normal vs C1/C6', 'Stat_Type': 'U-Test', 'Stat_Value': np.nan, 'P_Value': p_dpdi, 'Interpretation': 'Significant' if p_dpdi < 0.05 else 'Not Sig'})
    else:
        cprint("  > Insufficient data for conditional test.", f)

    cprint("\nB. Directional Test (Lagged Driver Analysis)", f)
    reg_df = df.dropna(subset=['is_Crash_Regime', 'PDI_lag1'])
    group_survive = reg_df[reg_df['is_Crash_Regime'] == 0]['PDI_lag1']
    group_crash = reg_df[reg_df['is_Crash_Regime'] == 1]['PDI_lag1']
    
    if len(group_survive) >= 5 and len(group_crash) >= 5:
        _, p_val_lag = stats.mannwhitneyu(group_survive, group_crash, alternative='two-sided')
        sig = "Leading Indicator (Pass)" if p_val_lag < 0.05 else "Concurrent Marker (Fail)"
        cprint(f"  > Lagged differences (Survival vs Crash): p-value = {p_val_lag:.4e} -> {sig}", f)
        if stats_list is not None:
            stats_list.append({'Scope': group_title, 'Test_Name': 'Directional Dynamics', 'Target': 'PDI_lag1', 'Comparison': 'Survive vs Crash', 'Stat_Type': 'U-Test', 'Stat_Value': np.nan, 'P_Value': p_val_lag, 'Interpretation': sig})

def test_robustness(df, group_title, f=None, stats_list=None):
    cprint("\n" + "="*80, f)
    cprint("6. ROBUSTNESS TESTS (TEMPORAL STABILITY & BOOTSTRAPPING)", f)
    cprint("="*80, f)
    
    cprint("A. Temporal Split (Pre-2020 vs Post-2020 Macro Shock)", f)
    df_pre = df[df['period_end'].dt.year < 2020]
    df_post = df[df['period_end'].dt.year >= 2020]
    
    def test_boundary(data, period_name):
        n_e3 = data[data['Configuration'] == 'Normal']['E_3'].dropna()
        s_e3 = data[data['Configuration'].isin(VALID_CONFIGS[1:])]['E_3'].dropna()
        if len(n_e3) > 3 and len(s_e3) > 3:
            _, p_val = stats.mannwhitneyu(n_e3, s_e3, alternative='two-sided')
            sig = "ROBUST" if p_val < 0.05 else "FAILED"
            cprint(f"  > E_3 Boundary ({period_name:10}): p-value = {p_val:.4e} -> {sig}", f)
            if stats_list is not None:
                stats_list.append({'Scope': group_title, 'Test_Name': 'Robustness Split', 'Target': 'E_3', 'Comparison': f'{period_name}: Normal vs Spec', 'Stat_Type': 'U-Test', 'Stat_Value': np.nan, 'P_Value': p_val, 'Interpretation': sig})
            
    test_boundary(df_pre, "Pre-2020")
    test_boundary(df_post, "Post-2020")

    if 'd_PDI_t' in df.columns:
        cprint("\nB. Bootstrapping (1,000 Iterations)", f)
        cond_mask = (df['R_t'] < 0.05) & (df['dK_Pi_prime_pct'] < 0)
        df_cond = df[cond_mask].dropna(subset=['d_PDI_t'])
        
        success_count = 0
        n_iterations = 1000
        for _ in range(n_iterations):
            sample_df = df_cond.sample(frac=0.8, replace=True)
            g_norm = sample_df[sample_df['Configuration'] == 'Normal']['d_PDI_t']
            g_crash = sample_df[sample_df['Configuration'].isin(['C1', 'C6'])]['d_PDI_t']
            
            if len(g_norm) > 2 and len(g_crash) > 2:
                if stats.mannwhitneyu(g_norm, g_crash, alternative='two-sided')[1] < 0.05:
                    success_count += 1
                    
        robust_pct = (success_count / n_iterations) * 100
        eval_text = "EXCELLENT (Robust against outliers)" if robust_pct > 90 else "WARNING (Sensitive to outliers)"
        cprint(f"  > Bootstrapping Success Rate (p < 0.05): {robust_pct:.1f}% -> {eval_text}", f)
        if stats_list is not None:
            stats_list.append({'Scope': group_title, 'Test_Name': 'Robustness Bootstrap', 'Target': 'd_PDI_t', 'Comparison': 'Normal vs C1/C6 (1000 iter)', 'Stat_Type': 'Success Rate %', 'Stat_Value': robust_pct, 'P_Value': np.nan, 'Interpretation': eval_text})
    cprint("="*80, f)

def run_all_tests(df_subset, group_title, f=None, stats_list=None, markov_list=None):
    """Wrapper function to execute all tests and collect metrics."""
    cprint("\n\n" + "#"*85, f)
    cprint(f"🚀 STATISTICAL SUITE: {group_title.upper()}", f)
    cprint(f"Observation Count: {len(df_subset):,} | Companies: {df_subset['Ticker'].nunique()}", f)
    cprint("#"*85, f)
    
    if len(df_subset) < 10:
        cprint("❌ Not enough data points to run statistical suite.", f)
        return
        
    analyze_markov_transitions(df_subset, group_title, f, markov_list)
    analyze_descriptive_pdi(df_subset, group_title, f, stats_list)
    analyze_correlations(df_subset, group_title, f, stats_list)
    analyze_kruskal_and_posthoc(df_subset, group_title, f, stats_list)
    test_pdi_dynamics(df_subset, group_title, f, stats_list)
    test_robustness(df_subset, group_title, f, stats_list)

def main():
    df = load_and_prep_data()
    if df is None: 
        return

    stats_export_list = []
    markov_export_list = []

    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        # 1. Run aggregate statistics for the global market
        run_all_tests(df, "GLOBAL MARKET", f, stats_export_list, markov_export_list)
        
        # 2. Run statistics individually for each sector
        if 'Sector' in df.columns:
            sectors = df['Sector'].dropna().unique()
            for sector in sorted(sectors):
                df_sector = df[df['Sector'] == sector].copy()
                run_all_tests(df_sector, f"SECTOR - {sector}", f, stats_export_list, markov_export_list)
        else:
            print("⚠️ Sector column not found. Skipping sector-specific analysis.")

    # 3. Export to CSV
    if stats_export_list:
        pd.DataFrame(stats_export_list).to_csv(CSV_STATS_FILE, index=False, encoding='utf-8-sig')
        print(f"✅ Comprehensive stats table exported to: {CSV_STATS_FILE}")
    
    if markov_export_list:
        pd.DataFrame(markov_export_list).to_csv(CSV_MARKOV_FILE, index=False, encoding='utf-8-sig')
        print(f"✅ Markov transition matrix exported to : {CSV_MARKOV_FILE}")

    print(f"✅ Text report exported to            : {REPORT_FILE}")

if __name__ == "__main__":
    main()
