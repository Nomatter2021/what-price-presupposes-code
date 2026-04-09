"""
15_PDI_Placebo_Tests.py

Academic Title:
    Placebo Tests for the PDI Leading Indicator:
    Within‑Firm Temporal Shuffling and Cross‑Firm Transfer

Objective:
    To assess whether the predictive power of PDI_t for next‑quarter crash
    (C1/C6) is genuinely temporal (i.e., PDI_t at the right time matters)
    or merely reflects persistent firm‑level characteristics.

    Two placebo approaches are implemented:

    Approach A (Within‑Firm Shuffle):
        - Keep crash labels (C1/C6 positions) fixed.
        - Shuffle PDI_t values randomly within each firm's time series.
        - Re‑run Mann‑Whitney test (crash_next vs safe_next).
        - If PDI timing is crucial, p‑value should become non‑significant.

    Approach B (Cross‑Firm Shuffle):
        - Within each sector, randomly pair firms.
        - Assign PDI_t sequence of firm A to firm B's timeline.
        - Keep crash labels of firm B unchanged.
        - Test whether PDI_t of firm A predicts crash of firm B.
        - If PDI is firm‑specific, p‑value should be ≈ 0.05 or higher.

Output Files (saved in data/results/):
    - 15_PDI_Placebo_Report.txt          : Comparison and interpretation.
    - 15_PDI_Placebo_Results.csv         : Detailed p‑values per iteration.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import mannwhitneyu
import logging
import warnings
from collections import defaultdict

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

TXT_REPORT = OUTPUT_DIR / 'report/15_PDI_Placebo_Report.txt'
CSV_RESULTS = OUTPUT_DIR / 'table/15_PDI_Placebo_Results.csv'

N_ITER = 200
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Valid states
CRASH_STATES = ['C1', 'C6']
EVOLVE_STATES = ['C3', 'C4']
EXCLUDED_SECTORS = ['Financials_and_Real_Estate', 'Financial']

# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================

def load_and_prepare():
    """Load final_panel.csv, apply formal gate, exclude Financials, compute necessary columns."""
    df = pd.read_csv(DATA_FILE)
    df['period_end'] = pd.to_datetime(df['period_end'])
    if 'Regime_Label' in df.columns:
        df['Configuration'] = np.where(df['Regime_Label'] == 'Normal_Regime', 'Normal', df['Configuration'])
    if 'Sector' in df.columns:
        df = df[~df['Sector'].isin(EXCLUDED_SECTORS)]
    # Keep only valid states (Normal, C1-C6)
    valid_states = ['Normal', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    df = df[df['Configuration'].isin(valid_states)].copy()
    df = df.sort_values(['Ticker', 'period_end']).reset_index(drop=True)
    # Ensure PDI_t is numeric
    if 'PDI_t' not in df.columns:
        raise KeyError("Column 'PDI_t' not found in data.")
    df['PDI_t'] = pd.to_numeric(df['PDI_t'], errors='coerce')
    # Create next crash indicator (for lag 1)
    group = df.groupby('Ticker')
    df['crash_next'] = group['Configuration'].shift(-1).isin(CRASH_STATES).astype(int)
    # We will use only C3/C4 quarters for the test
    df['is_c3c4'] = df['Configuration'].isin(EVOLVE_STATES)
    return df

def compute_pvalue(df, pdi_col='PDI_t'):
    """
    Compute Mann‑Whitney p‑value for PDI_t comparing:
        crash_next == 1 vs crash_next == 0
    restricted to C3/C4 rows.
    """
    c3c4 = df[df['is_c3c4']].dropna(subset=[pdi_col, 'crash_next'])
    if len(c3c4) == 0:
        return np.nan
    crash = c3c4[c3c4['crash_next'] == 1][pdi_col]
    safe = c3c4[c3c4['crash_next'] == 0][pdi_col]
    if len(crash) < 3 or len(safe) < 3:
        return np.nan
    _, p = mannwhitneyu(crash, safe, alternative='two-sided')
    return p

# ============================================================================
# APPROACH A: WITHIN‑FIRM SHUFFLE OF PDI_t
# ============================================================================

def approach_a_shuffle_pdi(df_orig, n_iter=N_ITER):
    """
    For each firm, keep crash labels fixed, shuffle PDI_t values within that firm.
    Recompute p-value each iteration.
    """
    results = []
    for i in range(n_iter):
        logger.debug(f"Approach A iteration {i+1}/{n_iter}")
        df_shuff = df_orig.copy()
        # Shuffle PDI_t within each firm
        for ticker, group in df_orig.groupby('Ticker'):
            indices = group.index
            pdi_vals = group['PDI_t'].values
            shuffled = np.random.permutation(pdi_vals)
            df_shuff.loc[indices, 'PDI_t'] = shuffled
        p_val = compute_pvalue(df_shuff, pdi_col='PDI_t')
        results.append({'iteration': i, 'p_value': p_val, 'approach': 'A_within_firm_shuffle'})
    return pd.DataFrame(results)

# ============================================================================
# APPROACH B: CROSS‑FIRM SHUFFLE (WITHIN SECTOR)
# ============================================================================

def approach_b_cross_firm_shuffle(df_orig, n_iter=N_ITER):
    """
    For each sector, randomly pair firms. For each pair, assign PDI_t sequence
    of firm A to firm B's timeline (keeping firm B's crash labels fixed).
    Then compute p-value using the swapped PDI_t.
    """
    results = []
    # Pre‑group by sector and ticker to get sequences
    # We need to work with firm‑level data structures
    sector_firms = defaultdict(list)
    for ticker, group in df_orig.groupby('Ticker'):
        sector = group['Sector'].iloc[0] if 'Sector' in group.columns else 'All'
        sector_firms[sector].append(ticker)

    for i in range(n_iter):
        logger.debug(f"Approach B iteration {i+1}/{n_iter}")
        df_swapped = df_orig.copy()
        # For each sector, randomly pair firms (if odd number, one firm is left out)
        for sector, tickers in sector_firms.items():
            if len(tickers) < 2:
                continue
            # Shuffle tickers list
            shuffled_tickers = np.random.permutation(tickers)
            # Pair consecutive firms
            for j in range(0, len(shuffled_tickers) - 1, 2):
                a = shuffled_tickers[j]
                b = shuffled_tickers[j+1]
                # Get PDI_t sequence of firm A (preserving order)
                seq_a = df_orig[df_orig['Ticker'] == a].sort_values('period_end')['PDI_t'].values
                # Get indices of firm B
                idx_b = df_orig[df_orig['Ticker'] == b].index
                # Assign PDI_t of A to B (if lengths match, else trim/pad? Ideally same length, but for safety)
                if len(seq_a) >= len(idx_b):
                    df_swapped.loc[idx_b, 'PDI_t'] = seq_a[:len(idx_b)]
                else:
                    # Pad with last value or random? Simpler: skip if unequal
                    continue
        p_val = compute_pvalue(df_swapped, pdi_col='PDI_t')
        results.append({'iteration': i, 'p_value': p_val, 'approach': 'B_cross_firm_shuffle'})
    return pd.DataFrame(results)

# ============================================================================
# MAIN
# ============================================================================

def main():
    logger.info("=" * 70)
    logger.info("PDI PLACEBO TESTS: WITHIN‑FIRM SHUFFLE & CROSS‑FIRM SHUFFLE")
    logger.info("=" * 70)

    # Load data
    logger.info("Loading data...")
    df = load_and_prepare()
    # Compute baseline p-value (real data, no shuffle)
    baseline_p = compute_pvalue(df)
    logger.info(f"Baseline p-value: {baseline_p:.4e}")

    # Approach A
    logger.info("\nRunning Approach A (within‑firm PDI shuffle)...")
    res_a = approach_a_shuffle_pdi(df, n_iter=N_ITER)
    median_a = res_a['p_value'].median()
    mean_a = res_a['p_value'].mean()
    pct_sig_a = (res_a['p_value'] < 0.05).mean() * 100
    logger.info(f"  Median p = {median_a:.4e}, Mean p = {mean_a:.4e}, % significant = {pct_sig_a:.1f}%")

    # Approach B
    logger.info("\nRunning Approach B (cross‑firm PDI shuffle)...")
    res_b = approach_b_cross_firm_shuffle(df, n_iter=N_ITER)
    median_b = res_b['p_value'].median()
    mean_b = res_b['p_value'].mean()
    pct_sig_b = (res_b['p_value'] < 0.05).mean() * 100
    logger.info(f"  Median p = {median_b:.4e}, Mean p = {mean_b:.4e}, % significant = {pct_sig_b:.1f}%")

    # Combine results
    all_res = pd.concat([res_a, res_b], ignore_index=True)
    all_res.to_csv(CSV_RESULTS, index=False)

    # Write report
    with open(TXT_REPORT, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("PDI PLACEBO TESTS REPORT\n")
        f.write("Approach A: Within‑firm shuffle of PDI_t (keep crash labels fixed)\n")
        f.write("Approach B: Cross‑firm shuffle of PDI_t (within sector, keep crash labels)\n")
        f.write("=" * 100 + "\n\n")

        f.write("I. BASELINE (Real Data)\n")
        f.write(f"   p-value = {baseline_p:.4e}\n")
        f.write(f"   Interpretation: {'Significant (p < 0.05)' if baseline_p < 0.05 else 'Not significant'}\n\n")

        f.write("II. APPROACH A – WITHIN‑FIRM SHUFFLE (200 iterations)\n")
        f.write(f"   Median p-value = {median_a:.4e}\n")
        f.write(f"   Mean p-value   = {mean_a:.4e}\n")
        f.write(f"   % of iterations with p < 0.05 = {pct_sig_a:.1f}%\n")
        f.write(f"   Interpretation: ")
        if pct_sig_a < 10:
            f.write("PDI timing matters – randomising PDI order destroys predictive power.\n")
        else:
            f.write("PDI effect persists even after shuffling – may be driven by firm‑level persistence.\n")
        f.write("\n")

        f.write("III. APPROACH B – CROSS‑FIRM SHUFFLE (200 iterations)\n")
        f.write(f"   Median p-value = {median_b:.4e}\n")
        f.write(f"   Mean p-value   = {mean_b:.4e}\n")
        f.write(f"   % of iterations with p < 0.05 = {pct_sig_b:.1f}%\n")
        f.write(f"   Interpretation: ")
        if pct_sig_b < 10:
            f.write("PDI signal is firm‑specific – PDI of one firm does not predict crash of another.\n")
        else:
            f.write("PDI effect is transferable across firms – may reflect sector‑wide conditions.\n")
        f.write("\n")

        f.write("IV. COMBINED INTERPRETATION\n")
        if baseline_p < 0.05 and pct_sig_a < 10 and pct_sig_b < 10:
            f.write("✅ Strong evidence: PDI_t is a genuine, temporally‑anchored, firm‑specific leading indicator.\n")
        elif baseline_p < 0.05 and pct_sig_a >= 10:
            f.write("⚠️ Weak evidence: PDI effect may be driven by persistent firm characteristics, not timing.\n")
        elif baseline_p < 0.05 and pct_sig_b >= 10:
            f.write("⚠️ Weak evidence: PDI effect may be sector‑wide, not firm‑specific.\n")
        else:
            f.write("❌ No evidence of leading indicator property.\n")

        f.write("\n" + "=" * 100 + "\n")
        f.write(f"Detailed results saved to {CSV_RESULTS}\n")

    logger.info(f"\n✅ Placebo tests completed. Report: {TXT_REPORT}")

if __name__ == "__main__":
    main()