"""
16_Diagnostics_Three_Tests.py (Corrected)

Three diagnostic tests:
1. PDI level vs change (Mann-Whitney)
2. Logistic regression with interaction (PDI_roll3 × dK_Pi_prime_pct)
3. Markov path dependency (permutation test on diagonal mean)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
import warnings
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_FILE = Path('../data/final_panel.csv')      # or out_of_sample.csv
OUTPUT_DIR = Path('results')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'report').mkdir(exist_ok=True)
(OUTPUT_DIR / 'table').mkdir(exist_ok=True)

TXT_REPORT = OUTPUT_DIR / 'report/16_Diagnostics_Report.txt'
CSV_TEST1 = OUTPUT_DIR / 'table/16_Test1_Results.csv'
CSV_TEST2 = OUTPUT_DIR / 'table/16_Test2_Logistic_Results.csv'
CSV_SURFACE = OUTPUT_DIR / 'table/16_Test2_Interaction_Surface.csv'

RANDOM_SEED = 42
N_PERM = 1000
np.random.seed(RANDOM_SEED)

# All possible states (including C5 which may have zero observations)
VALID_STATES = ['Normal', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
CRASH_STATES = ['C1', 'C6']
EVOLVE_STATES = ['C3', 'C4']
EXCLUDED_SECTORS = ['Financials_and_Real_Estate', 'Financial']

# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================

def load_and_prepare():
    df = pd.read_csv(DATA_FILE)
    df['period_end'] = pd.to_datetime(df['period_end'])
    if 'Regime_Label' in df.columns:
        df['Configuration'] = np.where(df['Regime_Label'] == 'Normal_Regime', 'Normal', df['Configuration'])
    if 'Sector' in df.columns:
        df = df[~df['Sector'].isin(EXCLUDED_SECTORS)]
    df = df[df['Configuration'].isin(VALID_STATES)].copy()
    df = df.sort_values(['Ticker', 'period_end']).reset_index(drop=True)

    # Ensure numeric columns
    for col in ['PDI_t', 'K_Pi_prime', 'E_3', 'PGR_t']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = np.nan

    # Compute ΔPDI and rolling means
    df['PDI_lag1'] = df.groupby('Ticker')['PDI_t'].shift(1)
    df['delta_PDI'] = df['PDI_t'] - df['PDI_lag1']
    df['PDI_roll3'] = df.groupby('Ticker')['PDI_t'].transform(lambda x: x.rolling(3, min_periods=2).mean())
    df['delta_PDI_roll3'] = df.groupby('Ticker')['delta_PDI'].transform(lambda x: x.rolling(3, min_periods=2).mean())

    # ΔK_Pi_prime (absolute change) – keep but we will use percentage
    df['K_Pi_prime_lag'] = df.groupby('Ticker')['K_Pi_prime'].shift(1)
    df['dK_Pi_prime'] = df['K_Pi_prime'] - df['K_Pi_prime_lag']
    # Percentage change (handle division by zero)
    df['dK_Pi_prime_pct'] = np.where(
        df['K_Pi_prime_lag'].notna() & (df['K_Pi_prime_lag'].abs() > 1e-9),
        df['dK_Pi_prime'] / df['K_Pi_prime_lag'].abs(),
        0.0
    )
    # Cap extreme values for stability (optional, but keep for robustness)
    df['dK_Pi_prime_pct'] = df['dK_Pi_prime_pct'].clip(-2, 2)

    # B = E_3 - (1 + PGR_t)
    if 'E_3' in df.columns and 'PGR_t' in df.columns:
        df['B'] = df['E_3'] - (1 + df['PGR_t'])
    else:
        df['B'] = np.nan

    # Next state and crash indicator
    df['next_config'] = df.groupby('Ticker')['Configuration'].shift(-1)
    df['crash_next'] = df['next_config'].isin(CRASH_STATES).astype(int)
    df['is_c3c4'] = df['Configuration'].isin(EVOLVE_STATES)

    return df

# ============================================================================
# TEST 1: PDI LEVEL VS CHANGE
# ============================================================================

def test1_pdi_variants(df):
    variants = {
        'PDI_t (level)': 'PDI_t',
        'ΔPDI_t (change)': 'delta_PDI',
        'PDI_roll3 (smooth level)': 'PDI_roll3',
        'ΔPDI_roll3 (smooth change)': 'delta_PDI_roll3'
    }
    results = []
    for name, col in variants.items():
        sub = df[df['is_c3c4']].dropna(subset=[col, 'crash_next'])
        if len(sub) == 0:
            continue
        crash_vals = sub[sub['crash_next'] == 1][col]
        safe_vals = sub[sub['crash_next'] == 0][col]
        if len(crash_vals) < 3 or len(safe_vals) < 3:
            continue
        u, p = mannwhitneyu(crash_vals, safe_vals, alternative='two-sided')
        n1, n2 = len(crash_vals), len(safe_vals)
        r = 1 - (2 * u) / (n1 * n2)   # rank-biserial correlation
        results.append({
            'Variant': name,
            'N_crash': n1,
            'N_safe': n2,
            'Median_crash': crash_vals.median(),
            'Median_safe': safe_vals.median(),
            'MannWhitney_U': u,
            'p_value': p,
            'Effect_size_r': r
        })
    df_res = pd.DataFrame(results)
    df_res.to_csv(CSV_TEST1, index=False)
    return df_res

# ============================================================================
# TEST 2: LOGISTIC REGRESSION WITH INTERACTION (USING PDI_roll3 AND dK_Pi_prime_pct)
# ============================================================================

def test2_logistic_interaction(df):
    # Use PDI_roll3 (smooth level) which was strongest in Test 1
    # and dK_Pi_prime_pct (percentage change, scale-invariant)
    predictors = ['PDI_roll3', 'dK_Pi_prime_pct', 'B']
    df_model = df[df['is_c3c4']].dropna(subset=predictors + ['crash_next']).copy()
    if len(df_model) == 0:
        return None, None, None

    # Prepare feature matrices
    X_base = df_model[['PDI_roll3']].values
    X_add = df_model[['PDI_roll3', 'dK_Pi_prime_pct']].values
    # Interaction term
    df_model['PDI_x_dK_pct'] = df_model['PDI_roll3'] * df_model['dK_Pi_prime_pct']
    X_inter = df_model[['PDI_roll3', 'dK_Pi_prime_pct', 'PDI_x_dK_pct']].values
    X_full = df_model[['PDI_roll3', 'dK_Pi_prime_pct', 'PDI_x_dK_pct', 'B']].values
    y = df_model['crash_next'].values

    models = {
        'Model1 (Baseline PDI_roll3)': X_base,
        'Model2 (Additive + dK_pct)': X_add,
        'Model3 (Interaction)': X_inter,
        'Model4 (Full + B)': X_full
    }
    results = []
    for name, X in models.items():
        clf = LogisticRegression(random_state=RANDOM_SEED, class_weight='balanced')
        clf.fit(X, y)
        y_proba = clf.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, y_proba)
        log_lik = -log_loss(y, y_proba) * len(y)
        coef_dict = {f'coef_{i}': c for i, c in enumerate(clf.coef_[0])}
        coef_dict['intercept'] = clf.intercept_[0]
        results.append({
            'Model': name,
            'AUC': auc,
            'LogLikelihood': log_lik,
            **coef_dict
        })
    df_res = pd.DataFrame(results)
    df_res.to_csv(CSV_TEST2, index=False)

    # Interaction surface for Model 3
    if len(df_res) >= 3:
        pdi_min, pdi_max = df_model['PDI_roll3'].min(), df_model['PDI_roll3'].max()
        dk_min, dk_max = df_model['dK_Pi_prime_pct'].min(), df_model['dK_Pi_prime_pct'].max()
        pdi_grid = np.linspace(pdi_min, pdi_max, 30)
        dk_grid = np.linspace(dk_min, dk_max, 30)
        grid_pdi, grid_dk = np.meshgrid(pdi_grid, dk_grid)
        coef3 = results[2]  # Model3 row
        intercept = coef3['intercept']
        coef_pdi = coef3['coef_0']
        coef_dk = coef3['coef_1']
        coef_inter = coef3['coef_2']
        log_odds = intercept + coef_pdi * grid_pdi + coef_dk * grid_dk + coef_inter * grid_pdi * grid_dk
        prob = 1 / (1 + np.exp(-log_odds))
        surface_df = pd.DataFrame({
            'PDI_roll3': grid_pdi.flatten(),
            'dK_Pi_prime_pct': grid_dk.flatten(),
            'Predicted_Prob_Crash': prob.flatten()
        })
        surface_df.to_csv(CSV_SURFACE, index=False)
    return df_res, surface_df if 'surface_df' in locals() else None, df_model

# ============================================================================
# TEST 3: MARKOV PATH DEPENDENCY (with handling of missing C5)
# ============================================================================

def test3_markov_path_dependency(df):
    # Extract sequences per firm
    seqs = df.groupby('Ticker')['Configuration'].agg(list).tolist()
    seqs = [seq for seq in seqs if len(seq) >= 2]

    # All possible states (including C5)
    states = VALID_STATES
    n_states = len(states)
    state_to_idx = {s: i for i, s in enumerate(states)}

    def compute_transition_matrix(seqs, state_to_idx, n_states):
        counts = np.zeros((n_states, n_states))
        for seq in seqs:
            for i in range(len(seq)-1):
                s = seq[i]
                t = seq[i+1]
                if s in state_to_idx and t in state_to_idx:
                    counts[state_to_idx[s], state_to_idx[t]] += 1
        row_sums = counts.sum(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            trans = np.divide(counts, row_sums, where=row_sums!=0)
            trans = np.nan_to_num(trans)
        return trans, counts

    # Observed transition matrix
    obs_trans, obs_counts = compute_transition_matrix(seqs, state_to_idx, n_states)

    # Marginal distribution (only over states that actually appear)
    all_config_flat = [cfg for seq in seqs for cfg in seq]
    marginal_full = pd.Series(all_config_flat).value_counts(normalize=True)
    marginal = pd.Series(0, index=states)
    marginal.update(marginal_full)

    # Expected transition matrix under iid (outer product of marginal)
    expected_trans = np.outer(marginal.values, marginal.values)

    # Compute mean diagonal of observed (only for states that have at least one occurrence)
    states_with_obs = [s for s in states if marginal[s] > 0]
    diag_vals = [obs_trans[state_to_idx[s], state_to_idx[s]] for s in states_with_obs]
    obs_diag_mean = np.mean(diag_vals)

    # Permutation test
    perm_diag_means = []
    for _ in range(N_PERM):
        shuffled_configs = np.random.permutation(all_config_flat)
        perm_seqs = []
        idx = 0
        for seq in seqs:
            length = len(seq)
            perm_seqs.append(shuffled_configs[idx:idx+length].tolist())
            idx += length
        perm_trans, _ = compute_transition_matrix(perm_seqs, state_to_idx, n_states)
        perm_diag = [perm_trans[state_to_idx[s], state_to_idx[s]] for s in states_with_obs]
        perm_diag_means.append(np.mean(perm_diag))
    perm_diag_means = np.array(perm_diag_means)
    p_value = np.mean(perm_diag_means >= obs_diag_mean)

    # Chi-square (only cells with expected count > 0)
    total_obs_len = len(all_config_flat)
    chi2_cells = []
    for i in range(n_states):
        for j in range(n_states):
            exp = expected_trans[i, j] * total_obs_len
            if exp > 0:
                obs = obs_counts[i, j]
                chi2_cells.append((obs - exp)**2 / exp)
    total_chi2 = np.sum(chi2_cells)

    diff = obs_trans - expected_trans

    return {
        'obs_diag_mean': obs_diag_mean,
        'perm_diag_mean_median': np.median(perm_diag_means),
        'p_value': p_value,
        'total_chi2': total_chi2,
        'states_with_obs': states_with_obs,
        'obs_trans': obs_trans,
        'expected_trans': expected_trans,
        'diff': diff
    }

# ============================================================================
# MAIN
# ============================================================================

def main():
    logger.info("=" * 70)
    logger.info("DIAGNOSTIC TESTS (Corrected - using PDI_roll3 and dK_Pi_prime_pct)")
    logger.info("=" * 70)

    df = load_and_prepare()
    logger.info(f"Data loaded: {len(df)} rows, {df['Ticker'].nunique()} firms")

    # Test 1
    logger.info("\n--- Test 1: PDI Level vs Change ---")
    res1 = test1_pdi_variants(df)
    logger.info(f"Results:\n{res1[['Variant', 'p_value', 'Effect_size_r']].to_string(index=False)}")

    # Test 2
    logger.info("\n--- Test 2: Logistic Regression (PDI_roll3 × dK_Pi_prime_pct) ---")
    res2, surface, df_model = test2_logistic_interaction(df)
    if res2 is not None:
        logger.info(f"Model AUCs:\n{res2[['Model', 'AUC']].to_string(index=False)}")
    else:
        logger.warning("Test 2 skipped (insufficient data)")

    # Test 3
    logger.info("\n--- Test 3: Markov Path Dependency ---")
    res3 = test3_markov_path_dependency(df)
    logger.info(f"Observed mean diagonal (states with obs): {res3['obs_diag_mean']:.4f}")
    logger.info(f"Permutation p-value (N={N_PERM}): {res3['p_value']:.4f}")
    logger.info(f"Total chi-square (observed vs iid): {res3['total_chi2']:.2f}")

    # Write report
    with open(TXT_REPORT, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("DIAGNOSTIC TESTS REPORT\n")
        f.write("Test 1: PDI Level vs Change (Mann-Whitney U)\n")
        f.write("Test 2: Logistic Regression with Interaction (PDI_roll3 × dK_Pi_prime_pct)\n")
        f.write("Test 3: Markov Path Dependency (Permutation test on diagonal mean)\n")
        f.write("=" * 100 + "\n\n")

        f.write("I. TEST 1 – PDI VARIANTS (C3/C4, crash_next=1 vs 0)\n")
        f.write(res1.to_string(index=False) + "\n\n")
        if not res1.empty:
            best = res1.loc[res1['p_value'].idxmin()]
            f.write(f"Best variant (lowest p): {best['Variant']} (p={best['p_value']:.2e}, r={best['Effect_size_r']:.3f})\n")
            f.write(f"Using PDI_roll3 for Test 2 as the strongest variant.\n\n")

        f.write("II. TEST 2 – LOGISTIC REGRESSION (C3/C4, crash_next as target)\n")
        f.write("Features: PDI_roll3 (smooth level), dK_Pi_prime_pct (percentage change, scale-invariant), B.\n")
        if res2 is not None:
            f.write(res2.to_string(index=False) + "\n\n")
            # Check interaction coefficient
            mod3 = res2[res2['Model'] == 'Model3 (Interaction)']
            if not mod3.empty and 'coef_2' in mod3.columns:
                coef_inter = mod3['coef_2'].iloc[0]
                f.write(f"Interaction coefficient (PDI_roll3 × dK_Pi_prime_pct) = {coef_inter:.4f}\n")
                if coef_inter < 0:
                    f.write("Interpretation: Negative interaction – low PDI_roll3 combined with high dK_pct strongly predicts crash.\n")
                else:
                    f.write("Interpretation: Positive interaction – not as expected.\n")
        else:
            f.write("Insufficient data for logistic regression.\n\n")

        f.write("III. TEST 3 – MARKOV PATH DEPENDENCY\n")
        f.write(f"Number of permutations: {N_PERM}\n")
        f.write(f"States with observations: {res3['states_with_obs']}\n")
        f.write(f"Observed mean diagonal probability: {res3['obs_diag_mean']:.4f}\n")
        f.write(f"Median permuted diagonal mean: {res3['perm_diag_mean_median']:.4f}\n")
        f.write(f"Permutation p-value: {res3['p_value']:.4f}\n")
        f.write(f"Total chi-square (observed vs iid expected): {res3['total_chi2']:.2f}\n")
        if res3['p_value'] < 0.05:
            f.write("Conclusion: Observed diagonal is significantly larger than expected under iid → genuine path dependency.\n")
        else:
            f.write("Conclusion: No evidence of path dependency beyond marginal distribution.\n")

        f.write("\n" + "=" * 100 + "\n")
        f.write(f"Output CSV files:\n - Test1: {CSV_TEST1}\n - Test2: {CSV_TEST2}\n - Interaction surface: {CSV_SURFACE}\n")

    logger.info(f"\n✅ All tests completed. Report: {TXT_REPORT}")

if __name__ == "__main__":
    main()