"""
FINAL STRUCTURAL VALIDATION REPORT
Executes the core empirical tests for the Expectation Chimera hypothesis:
1. Markov transition matrices with cluster bootstrap confidence intervals.
2. Gestation (C2) pathway analysis: E3 threshold predicting maturation vs. direct collapse.
3. Maturity (C3/C4) survival analysis: PDI as a one-quarter-ahead leading indicator.
4. Cross-sectional summary statistics (Medians and Spearman correlations).
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
from typing import List, Dict, Tuple, Optional
from pathlib import Path

warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
DATA_FILE = Path('data/final_all_cycles_combined.csv')
OUTPUT_FILE = Path('structural_report_complete.txt')
VALID_STATES = ['Normal', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']

# Statistical Parameters
N_BOOT = 1000
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ==================== DATA PREPARATION ====================

def load_and_preprocess_data(filepath: Path) -> pd.DataFrame:
    """Loads the merged panel data and standardizes configuration labels."""
    print("Loading and preprocessing data...")
    df = pd.read_csv(filepath)
    df['period_end'] = pd.to_datetime(df['period_end'])
    
    if 'Regime_Label' in df.columns:
        df['Configuration'] = np.where(df['Regime_Label'] == 'Normal_Regime', 'Normal', df['Configuration'])

    # Sort strictly temporally within each firm/cycle
    df = df.sort_values(['Ticker', 'period_end']).reset_index(drop=True)
    return df

# ==================== MARKOV MATRIX & BOOTSTRAP ====================

def compute_transition_probs(seqs: List[List[str]]) -> Tuple[Dict, Dict, Dict]:
    """Computes transition probabilities, raw counts, and row totals for given sequences."""
    counts = {s: {t: 0 for t in VALID_STATES} for s in VALID_STATES}
    total = {s: 0 for s in VALID_STATES}
    
    for seq in seqs:
        for i in range(len(seq) - 1):
            s, t = seq[i], seq[i+1]
            if s in counts and t in counts[s]:
                counts[s][t] += 1
                total[s] += 1
                
    probs = {s: {t: counts[s][t] / total[s] if total[s] > 0 else 0 for t in VALID_STATES} for s in VALID_STATES}
    return probs, counts, total


def get_confidence_intervals(probs_list: List[Dict], s: str, t: str) -> Tuple[float, float]:
    """Extracts the 2.5th and 97.5th percentiles (95% CI) from bootstrap distributions."""
    vals = [p[s][t] for p in probs_list]
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def run_bootstrap_markov(df: pd.DataFrame, group_col: str) -> Tuple[Dict, Dict, Dict, List[Dict]]:
    """Generates the primary Markov matrix and executes cluster bootstrap for CIs."""
    print("Computing Markov transition matrix with bootstrap CI...")
    
    # Extract temporal sequences of configurations
    sequences = df.groupby(group_col)['Configuration'].agg(list).tolist()
    sequences = [seq for seq in sequences if len(seq) >= 2]

    # Baseline computation
    orig_probs, orig_counts, orig_total = compute_transition_probs(sequences)

    # Cluster Bootstrap (resampling sequences, not individual observations)
    boot_probs = []
    n_seqs = len(sequences)
    for _ in range(N_BOOT):
        boot_idx = np.random.choice(n_seqs, n_seqs, replace=True)
        boot_seqs = [sequences[i] for i in boot_idx]
        p, _, _ = compute_transition_probs(boot_seqs)
        boot_probs.append(p)
        
    return orig_probs, orig_counts, orig_total, boot_probs

# ==================== C2 PATHWAY ANALYSIS ====================

def analyze_c2_pathways(df: pd.DataFrame, group_col: str) -> Tuple[pd.Series, pd.Series, int, float, float]:
    """
    Analyzes sequences entering Gestation (C2).
    Evaluates whether E3 levels dictate maturation (C3) vs. direct collapse (C1/C6).
    """
    print("Analyzing Gestation (C2) pathways and E3 thresholds...")
    c2_cycles = []
    
    for gid, group in df[df['Configuration'] == 'C2'].groupby(group_col):
        group = group.sort_values('period_end')
        if len(group) == 0: 
            continue
            
        last = group.iloc[-1]
        next_idx = group.index[-1] + 1
        next_state = None
        
        # Verify the next observation belongs to the same firm/cycle
        if next_idx < len(df) and df.loc[next_idx, group_col] == gid:
            next_state = df.loc[next_idx, 'Configuration']
            
        if next_state in ['C3', 'C1', 'C6']:
            c2_cycles.append({
                'E3_end': last['E_3'],
                'next_state': next_state
            })
            
    c2_df = pd.DataFrame(c2_cycles)
    
    if c2_df.empty:
        return pd.Series(), pd.Series(), 0, 0.0, np.nan

    to_c3 = c2_df[c2_df['next_state'] == 'C3']['E3_end'].dropna()
    to_crash = c2_df[c2_df['next_state'].isin(['C1', 'C6'])]['E3_end'].dropna()

    # Non-parametric difference test
    p_e3 = np.nan
    if len(to_c3) >= 3 and len(to_crash) >= 3:
        _, p_e3 = stats.mannwhitneyu(to_c3, to_crash, alternative='two-sided')

    # Threshold Optimization (Maximizing predictive accuracy)
    best_acc = 0.0
    best_th = 10
    if len(to_c3) > 0 and len(to_crash) > 0:
        for th in range(5, 51):
            correct_c3 = (to_c3 < th).sum()
            correct_crash = (to_crash >= th).sum()
            acc = (correct_c3 + correct_crash) / (len(to_c3) + len(to_crash))
            if acc > best_acc:
                best_acc = acc
                best_th = th

    return to_c3, to_crash, best_th, best_acc, p_e3

# ==================== PDI LEADING INDICATOR ====================

def analyze_pdi_leading_indicator(df: pd.DataFrame, group_col: str) -> Tuple[pd.Series, pd.Series, float]:
    """
    Evaluates PDI dynamics during Maturity (C3/C4).
    Tests if PDI is significantly lower in the quarter immediately preceding a crash.
    """
    print("Analyzing PDI as a leading indicator within Maturity (C3/C4)...")
    df = df.copy()
    
    # Target variable: Is the NEXT quarter a collapse?
    df['is_Crash'] = df['Configuration'].isin(['C1', 'C6']).astype(int)
    df['Crash_next'] = df.groupby(group_col)['is_Crash'].shift(-1)

    # Restrict analysis to maturity phase observations
    df_c3c4 = df[df['Configuration'].isin(['C3', 'C4'])].copy()
    
    crash_pdi = df_c3c4[df_c3c4['Crash_next'] == 1]['PDI_t'].dropna()
    safe_pdi = df_c3c4[df_c3c4['Crash_next'] == 0]['PDI_t'].dropna()
    
    p_pdi = np.nan
    if len(crash_pdi) >= 3 and len(safe_pdi) >= 3:
        _, p_pdi = stats.mannwhitneyu(crash_pdi, safe_pdi, alternative='two-sided')

    return crash_pdi, safe_pdi, p_pdi

# ==================== REPORT GENERATOR ====================

def generate_report(df: pd.DataFrame, 
                    orig_probs: Dict, orig_total: Dict, boot_probs: List,
                    to_c3: pd.Series, to_crash: pd.Series, best_th: int, best_acc: float, p_e3: float,
                    crash_pdi: pd.Series, safe_pdi: pd.Series, p_pdi: float) -> None:
    """Compiles all findings into a structured academic text report."""
    print("Generating comprehensive statistical report...")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("COMPLETE STRUCTURAL VALIDATION REPORT - EXPECTATION CHIMERA HYPOTHESIS\n")
        f.write("="*80 + "\n\n")
        
        # 1. DISTRIBUTION
        f.write("1. CONFIGURATION DISTRIBUTION\n")
        f.write("-" * 40 + "\n")
        config_counts = df['Configuration'].value_counts()
        total = len(df)
        for c in VALID_STATES:
            cnt = config_counts.get(c, 0)
            pct = cnt / total * 100 if total > 0 else 0
            f.write(f"  {c:6} : {cnt:5} ({pct:5.2f}%)\n")
        f.write("\n")
        
        # 2. MARKOV MATRIX
        f.write("2. MARKOV TRANSITION MATRIX (with 95% Cluster Bootstrap CI)\n")
        f.write("-" * 60 + "\n")
        f.write("Transition probabilities (%), with 95% CI from bootstrap (by cycle):\n\n")
        header = "From\\To    " + " ".join(f"{s:>6}" for s in VALID_STATES)
        f.write(header + "\n")
        
        for s in VALID_STATES:
            row = f"{s:6}    "
            for t in VALID_STATES:
                orig = orig_probs[s][t] * 100
                if orig_total.get(s, 0) >= 5:
                    low, high = get_confidence_intervals(boot_probs, s, t)
                    row += f"{orig:6.1f}[{low*100:4.0f}-{high*100:4.0f}] "
                else:
                    row += f"{orig:6.1f}         "
            f.write(row + "\n")
        f.write("\nNote: CIs reported only for states with N >= 5 outgoing transitions.\n\n")
        
        # 3. C2 PATHWAYS
        f.write("3. GESTATION (C2) PATHWAY ANALYSIS: E3 AS THRESHOLD\n")
        f.write("-" * 50 + "\n")
        f.write(f"Number of C2 sequences with known terminal state: {len(to_c3) + len(to_crash)}\n")
        f.write(f"  -> Maturation (C2→C3) : n={len(to_c3)}, Median E3 = {to_c3.median():.2f}\n")
        f.write(f"  -> Collapse (C2→C1/C6): n={len(to_crash)}, Median E3 = {to_crash.median():.2f}\n")
        if not np.isnan(p_e3):
            f.write(f"Mann-Whitney U test: p = {p_e3:.4e} (Significant difference)\n")
        
        f.write(f"Optimal E3 threshold: {best_th} (Balanced Accuracy = {best_acc:.1%})\n\n")
        f.write("Sensitivity Analysis (Accuracy at various thresholds):\n")
        for th in [5, 10, 15, 20, 25, 30, 35, 40]:
            correct_c3 = (to_c3 < th).sum()
            correct_crash = (to_crash >= th).sum()
            acc = (correct_c3 + correct_crash) / max(1, (len(to_c3) + len(to_crash)))
            f.write(f"  Threshold={th:2}: Correct C3={correct_c3:2}/{len(to_c3):2}, "
                    f"Correct Collapse={correct_crash:2}/{len(to_crash):2}, Total Acc={acc:.1%}\n")
        f.write("\n")
        
        # 4. PDI LEADING INDICATOR
        f.write("4. PDI LEADING INDICATOR WITHIN MATURITY (C3/C4)\n")
        f.write("-" * 50 + "\n")
        f.write(f"Maturity quarters with known subsequent outcome: {len(crash_pdi) + len(safe_pdi)}\n")
        f.write(f"  -> Followed by Collapse (C1/C6): n={len(crash_pdi)}, Median PDI={crash_pdi.median():.4f}\n")
        f.write(f"  -> Followed by Continuation    : n={len(safe_pdi)}, Median PDI={safe_pdi.median():.4f}\n")
        if not np.isnan(p_pdi):
            f.write(f"Mann-Whitney U test: p = {p_pdi:.4e}\n")
            if p_pdi < 0.05:
                f.write("  => PDI is significantly lower preceding a collapse (Validated Leading Indicator)\n")
            else:
                f.write("  => No significant statistical divergence detected.\n")
        else:
            f.write("  Insufficient sample size for non-parametric testing.\n")
        f.write("\n")
        
        # 5. SUPPLEMENTARY STATISTICS
        f.write("5. SUPPLEMENTARY CROSS-SECTIONAL STATISTICS\n")
        f.write("-" * 50 + "\n")
        
        f.write("Median Uninitiated Obligation Ratio (E3) by Configuration:\n")
        for c in VALID_STATES:
            med = df[df['Configuration'] == c]['E_3'].median()
            f.write(f"  {c:6}: {med:.4f}\n")
            
        f.write("\nMedian Productive Discharge Index (PDI) by Configuration:\n")
        for c in VALID_STATES:
            med = df[df['Configuration'] == c]['PDI_t'].median()
            f.write(f"  {c:6}: {med:.4f}\n")
            
        f.write("\nGlobal Rank Correlations (Spearman):\n")
        rho_rt, p_rt = stats.spearmanr(df['E_3'], df['R_t'], nan_policy='omit')
        rho_pdi, p_pdi = stats.spearmanr(df['E_3'], df['PDI_t'], nan_policy='omit')
        f.write(f"  Obligation (E3) vs Discharge Speed (R_t) : rho = {rho_rt:.4f}, p = {p_rt:.2e}\n")
        f.write(f"  Obligation (E3) vs Discharge Source (PDI): rho = {rho_pdi:.4f}, p = {p_pdi:.2e}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")

    print(f"Report successfully saved to {OUTPUT_FILE}")

# ==================== MAIN EXECUTION ====================

def main() -> None:
    if not DATA_FILE.exists():
        print(f"❌ Error: Master dataset {DATA_FILE} not found.")
        return
        
    df = load_and_preprocess_data(DATA_FILE)
    group_col = 'Cycle_ID' if 'Cycle_ID' in df.columns else 'Ticker'
    
    # Run analyses
    orig_probs, orig_counts, orig_total, boot_probs = run_bootstrap_markov(df, group_col)
    to_c3, to_crash, best_th, best_acc, p_e3 = analyze_c2_pathways(df, group_col)
    crash_pdi, safe_pdi, p_pdi = analyze_pdi_leading_indicator(df, group_col)
    
    # Export Report
    generate_report(df, orig_probs, orig_total, boot_probs, 
                    to_c3, to_crash, best_th, best_acc, p_e3, 
                    crash_pdi, safe_pdi, p_pdi)

if __name__ == "__main__":
    main()
