"""
05_Financials_normal_impact_test.py

Academic Title:
    Testing the Impact of the 'Normal' State on the C1-C6 Transition Matrix
    within the Financials Sector

Objective:
    This script examines whether the presence of the 'Normal' state in the
    Financials sector alters the estimated transition probabilities among the
    six crisis states (C1–C6). The analysis addresses the concern that 'Normal'
    may act as a structural outlier, thereby distorting the Markov dynamics
    when only C1–C6 are of interest.

Methodology:
    1. Extract all state sequences from the Financials sector, including 'Normal'.
    2. Create a "compressed" version of these sequences by removing all
       'Normal' observations while preserving the order of C1–C6.
    3. Compare the C1–C6 transition matrix derived from:
         (a) Sequences that never included 'Normal' (i.e., filtered from the start).
         (b) Compressed sequences (i.e., after stripping 'Normal').
    4. Statistical inference:
         - Bootstrap (N = 1,000) to estimate 95% confidence intervals for each
           transition probability.
         - Permutation test (N = 1,000) to assess the overall distance between
           the two matrices (Euclidean distance).

Output Files:
    - financials_normal_impact_report.txt   : Detailed academic report.
    - financials_normal_impact_comparison.csv : Comparative table of transition
                                               probabilities with CIs.

Dependencies:
    pandas, numpy, scipy, pathlib

Author: (Generated for research purposes)
Date: (current)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_FILE = Path('../data/final_panel.csv')
OUTPUT_DIR = Path('results')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TXT_REPORT = OUTPUT_DIR / 'report/financials_normal_impact_report.txt'
CSV_COMPARE = OUTPUT_DIR / 'table/financials_normal_impact_comparison.csv'

VALID_STATES = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']
ALL_STATES = ['Normal'] + VALID_STATES
RANDOM_SEED = 42
N_BOOT = 1000
N_PERM = 1000

np.random.seed(RANDOM_SEED)


# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================

def load_financials_data():
    """
    Load the panel dataset and filter for the Financials sector.
    Ensures the 'Normal' state is consistently labelled.
    """
    df = pd.read_csv(DATA_FILE)
    df['period_end'] = pd.to_datetime(df['period_end'])
    
    # Apply the formal gate: override Configuration based on Regime_Label
    if 'Regime_Label' in df.columns:
        df['Configuration'] = np.where(
            df['Regime_Label'] == 'Normal_Regime',
            'Normal',
            df['Configuration']
        )
    
    if 'Sector' not in df.columns:
        raise ValueError("The dataset does not contain a 'Sector' column.")
    
    # Subset to Financials sector and valid states
    df_fin = df[df['Sector'] == 'Financials_and_Real_Estate'].copy()
    df_fin = df_fin[df_fin['Configuration'].isin(ALL_STATES)].copy()
    df_fin = df_fin.sort_values(['Ticker', 'period_end']).reset_index(drop=True)
    
    # Determine grouping column: prefer Cycle_ID, else Ticker
    group_col = 'Cycle_ID' if 'Cycle_ID' in df_fin.columns else 'Ticker'
    return df_fin, group_col


def extract_sequences(df, group_col, allowed_states):
    """
    Extract state sequences per group (firm or cycle) after filtering to
    a given set of allowed states. Only sequences of length >= 2 are retained.
    """
    df_filtered = df[df['Configuration'].isin(allowed_states)].copy()
    sequences = df_filtered.groupby(group_col)['Configuration'].agg(list).tolist()
    return [seq for seq in sequences if len(seq) >= 2]


def compress_sequences(sequences, state_to_remove='Normal'):
    """
    Remove all occurrences of a specified state (e.g., 'Normal') from each
    sequence while preserving the relative order of the remaining states.
    Only compressed sequences with length >= 2 are kept.
    """
    compressed = []
    for seq in sequences:
        new_seq = [s for s in seq if s != state_to_remove]
        if len(new_seq) >= 2:
            compressed.append(new_seq)
    return compressed


# ============================================================================
# MARKOV TRANSITION PROBABILITIES
# ============================================================================

def compute_transition_probs(sequences, states):
    """
    Compute the first-order Markov transition probability matrix.
    Returns:
        probs  : dict of dict (from-state -> to-state -> probability)
        counts : dict of dict (raw transition counts)
        totals : dict (total outgoing counts per from-state)
    """
    counts = {s: {t: 0 for t in states} for s in states}
    totals = {s: 0 for s in states}
    
    for seq in sequences:
        for i in range(len(seq) - 1):
            s, t = seq[i], seq[i + 1]
            if s in counts and t in counts[s]:
                counts[s][t] += 1
                totals[s] += 1
    
    probs = {}
    for s in states:
        probs[s] = {}
        for t in states:
            if totals[s] > 0:
                probs[s][t] = counts[s][t] / totals[s]
            else:
                probs[s][t] = 0.0
    return probs, counts, totals


def bootstrap_confidence_intervals(sequences, states, n_iter=N_BOOT):
    """
    Generate bootstrap replicates of the transition probability matrix
    by resampling sequences with replacement.
    Returns a list of bootstrap matrices (each as a nested dict).
    """
    boot_samples = []
    n_seq = len(sequences)
    for _ in range(n_iter):
        boot_seqs = [sequences[i] for i in np.random.choice(n_seq, n_seq, replace=True)]
        p, _, _ = compute_transition_probs(boot_seqs, states)
        boot_samples.append(p)
    return boot_samples


def extract_ci_from_bootstrap(boot_samples, from_state, to_state):
    """
    Extract the 2.5th and 97.5th percentiles for a specific transition
    probability from the bootstrap distribution.
    """
    vals = [p[from_state][to_state] for p in boot_samples]
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


# ============================================================================
# PERMUTATION TEST FOR MATRIX EQUALITY
# ============================================================================

def euclidean_distance_matrix(p1, p2, states):
    """
    Compute the squared Euclidean distance between two transition matrices.
    """
    dist = 0.0
    for s in states:
        for t in states:
            dist += (p1[s][t] - p2[s][t]) ** 2
    return dist


def permutation_test(seqs_a, seqs_b, states, n_perm=N_PERM):
    """
    Permutation test for the null hypothesis that the two sets of sequences
    come from the same transition process. The test statistic is the Euclidean
    distance between the two estimated matrices.
    Returns:
        observed_distance : float
        p_value            : float
    """
    # Observed distance
    p_a, _, _ = compute_transition_probs(seqs_a, states)
    p_b, _, _ = compute_transition_probs(seqs_b, states)
    obs_dist = euclidean_distance_matrix(p_a, p_b, states)
    
    # Pool and permute
    combined = seqs_a + seqs_b
    n_a = len(seqs_a)
    perm_dists = []
    
    for _ in range(n_perm):
        np.random.shuffle(combined)
        perm_a = combined[:n_a]
        perm_b = combined[n_a:]
        p_perm_a, _, _ = compute_transition_probs(perm_a, states)
        p_perm_b, _, _ = compute_transition_probs(perm_b, states)
        perm_dists.append(euclidean_distance_matrix(p_perm_a, p_perm_b, states))
    
    p_val = np.mean(np.array(perm_dists) >= obs_dist)
    return obs_dist, p_val


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 80)
    print("ACADEMIC TEST: IMPACT OF 'NORMAL' STATE ON C1-C6 TRANSITION MATRIX")
    print("SECTOR: FINANCIALS")
    print("=" * 80)
    
    # 1. Load data
    df_fin, group_col = load_financials_data()
    print(f"Loaded {len(df_fin):,} observations from Financials sector.")
    
    # 2. Extract sequences under different filters
    seqs_with_normal = extract_sequences(df_fin, group_col, ALL_STATES)
    seqs_without_normal = extract_sequences(df_fin, group_col, VALID_STATES)
    seqs_compressed = compress_sequences(seqs_with_normal, state_to_remove='Normal')
    
    print(f"  - Sequences including 'Normal'          : {len(seqs_with_normal)}")
    print(f"  - Sequences excluding 'Normal' (raw)    : {len(seqs_without_normal)}")
    print(f"  - Compressed sequences (after stripping): {len(seqs_compressed)}")
    
    # 3. Bootstrap CI for both approaches
    boot_without = bootstrap_confidence_intervals(seqs_without_normal, VALID_STATES)
    boot_compressed = bootstrap_confidence_intervals(seqs_compressed, VALID_STATES)
    
    # 4. Point estimates
    p_without, _, _ = compute_transition_probs(seqs_without_normal, VALID_STATES)
    p_compressed, _, _ = compute_transition_probs(seqs_compressed, VALID_STATES)
    
    # 5. Permutation test
    obs_dist, p_val = permutation_test(seqs_without_normal, seqs_compressed, VALID_STATES)
    
    # 6. Build comparison table
    rows = []
    for s in VALID_STATES:
        for t in VALID_STATES:
            prob_wo = p_without[s][t] * 100
            ci_wo_low, ci_wo_high = extract_ci_from_bootstrap(boot_without, s, t)
            ci_wo_low *= 100
            ci_wo_high *= 100
            
            prob_c = p_compressed[s][t] * 100
            ci_c_low, ci_c_high = extract_ci_from_bootstrap(boot_compressed, s, t)
            ci_c_low *= 100
            ci_c_high *= 100
            
            overlap = not (ci_wo_high < ci_c_low or ci_c_high < ci_wo_low)
            
            rows.append({
                'From_State': s,
                'To_State': t,
                'Without_Normal_Prob_Pct': prob_wo,
                'Without_Normal_CI95_Lower': ci_wo_low,
                'Without_Normal_CI95_Upper': ci_wo_high,
                'Compressed_Prob_Pct': prob_c,
                'Compressed_CI95_Lower': ci_c_low,
                'Compressed_CI95_Upper': ci_c_high,
                'CI_Overlap': overlap
            })
    
    df_compare = pd.DataFrame(rows)
    
    # 7. Write academic report
    with open(TXT_REPORT, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("REPORT: IMPACT OF THE 'NORMAL' STATE ON C1-C6 TRANSITION PROBABILITIES\n")
        f.write("SECTOR: FINANCIALS AND REAL ESTATE\n")
        f.write("=" * 100 + "\n\n")
        
        f.write("1. SAMPLE DESCRIPTION\n")
        f.write(f"   - Number of sequences containing 'Normal': {len(seqs_with_normal)}\n")
        f.write(f"   - Number of sequences without 'Normal' (direct filter): {len(seqs_without_normal)}\n")
        f.write(f"   - Number of compressed sequences (post-removal of 'Normal'): {len(seqs_compressed)}\n\n")
        
        f.write("2. PERMUTATION TEST (EQUALITY OF TRANSITION MATRICES)\n")
        f.write(f"   Null hypothesis: The C1-C6 transition dynamics are unaffected by\n")
        f.write(f"                    the presence/removal of 'Normal' states.\n")
        f.write(f"   Test statistic: Euclidean distance between matrices.\n")
        f.write(f"   Observed distance = {obs_dist:.6f}\n")
        f.write(f"   Permutation p-value = {p_val:.4f}  (based on {N_PERM} permutations)\n")
        if p_val < 0.05:
            f.write("   → Conclusion: Reject H0. The removal of 'Normal' significantly alters\n")
            f.write("     the C1-C6 transition matrix. 'Normal' acts as a structural outlier\n")
            f.write("     within the Financials sector.\n")
        else:
            f.write("   → Conclusion: Fail to reject H0. No sufficient evidence that\n")
            f.write("     'Normal' influences the C1-C6 transition dynamics.\n")
        f.write("\n")
        
        f.write("3. BOOTSTRAP CONFIDENCE INTERVALS (SELECTED TRANSITIONS)\n")
        f.write("   (95% percentile intervals, 1,000 bootstrap replications)\n")
        f.write("-" * 80 + "\n")
        f.write("   From → To | Without Normal (CI)         | Compressed (CI)            | Overlap\n")
        f.write("-" * 80 + "\n")
        # Show a few illustrative transitions
        for s in ['C2', 'C3']:
            for t in ['C3', 'C1', 'C6']:
                row = df_compare[(df_compare['From_State'] == s) & (df_compare['To_State'] == t)]
                if not row.empty:
                    r = row.iloc[0]
                    f.write(f"   {s} → {t}    | {r['Without_Normal_Prob_Pct']:5.1f}%  "
                            f"[{r['Without_Normal_CI95_Lower']:3.0f}-{r['Without_Normal_CI95_Upper']:3.0f}]   | "
                            f"{r['Compressed_Prob_Pct']:5.1f}%  [{r['Compressed_CI95_Lower']:3.0f}-{r['Compressed_CI95_Upper']:3.0f}]   | "
                            f"{'Yes' if r['CI_Overlap'] else 'No'}\n")
        f.write("\n")
        
        f.write("4. FULL RESULTS\n")
        f.write(f"   Complete comparison table with all transitions is available in:\n")
        f.write(f"   {CSV_COMPARE.name}\n\n")
        f.write("=" * 100 + "\n")
    
    # 8. Export CSV
    df_compare.to_csv(CSV_COMPARE, index=False, encoding='utf-8-sig')
    
    print("\n" + "=" * 60)
    print(f"✓ Analysis completed.")
    print(f"  - Academic report: {TXT_REPORT}")
    print(f"  - Comparison table: {CSV_COMPARE}")
    print("=" * 60)


if __name__ == '__main__':
    main()