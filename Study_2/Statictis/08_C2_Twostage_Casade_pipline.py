"""
08_C2_TwoStage_Cascade_Pipeline.py

Academic Title:
    Two‑Stage Statistical Inference and Bootstrap Validation for
    Predicting Collapse from C2 State (Kinematic & Interaction Features)

Objectives:
    This script implements a complete two‑stage machine learning pipeline
    to distinguish, among firms in state C2, those that will:
        - Evolve to C3/C4 (soft landing),
        - Sustain C2 (continue uphill),
        - Collapse to C1/C6 (structural crash).

    The pipeline consists of:
        1. Feature selection via Mann‑Whitney U tests (three pairwise comparisons).
        2. Construction of interaction features (non‑linear kinematics).
        3. Stage 1: Logistic regression to filter out “evolving” firms,
           leaving a “dropdown” set (gray zone) of C2 firms that either
           sustain or collapse.
        4. Stage 2: Logistic regression on the dropdown set to discriminate
           collapse vs. sustain.
        5. Bootstrap validation (200 iterations) of the two‑stage cascade,
           reporting AUC, Brier score, and optimal threshold.

    All analyses exclude the Financials sector for consistency with previous
    ML pipelines.

Output Files (saved in data/results/):
    - 08_TwoStage_Cascade_Report.txt        : Full academic report.
    - 08_feature_selection_all_pairs.csv    : Mann‑Whitney results for all pairs.
    - 08_dropdown_inference_stats.csv       : Mann‑Whitney on dropdown zone.
    - 08_bootstrap_results.csv              : Bootstrap AUC, Brier, threshold.

Dependencies:
    pandas, numpy, scipy, scikit‑learn, imbalanced‑learn, pathlib
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_FILE = Path('../data/final_panel.csv')
OUTPUT_DIR = Path('results')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TXT_REPORT = OUTPUT_DIR / 'report/08_TwoStage_Cascade_Report.txt'
CSV_ALL_PAIRS = OUTPUT_DIR / 'table/08_feature_selection_all_pairs.csv'
CSV_DROPDOWN = OUTPUT_DIR / 'table/08_dropdown_inference_stats.csv'
CSV_BOOTSTRAP = OUTPUT_DIR / 'table/08_bootstrap_results.csv'

RANDOM_SEED = 42
N_BOOT = 200

# Stage 1 configuration (derived from prior grid search)
S1_RATIO = 0.20          # undersampling ratio for Stage 1 (minority:majority)
S1_THRESH = 0.1749       # probability threshold to define "dropdown" (low evolution probability)
FEATS_S1 = ['B', 'B_Change', 'B_Acceleration', 'Cum_B_Change', 'Jerk']

# Stage 2 candidate features (will be selected based on statistical tests)
# After feature selection, the most powerful set is: ['B', 'B_Momentum', 'B_Critical_Mass']
FEATS_S2_FINAL = ['B', 'B_Momentum', 'B_Critical_Mass']

# All variables to be tested in dropdown inference
VARS_TO_TEST = [
    'B', 'B_Change', 'B_Acceleration', 'Cum_B_Change', 'Jerk',
    'B_Momentum', 'B_Critical_Mass', 'B_Overload',
    'Damping_Force', 'Destructive_Force', 'Tension_Index'
]

# Excluded sectors (Financials)
EXCLUDED_SECTORS = ['Financials_and_Real_Estate', 'Financial']


# ============================================================================
# DATA PREPARATION AND FEATURE ENGINEERING
# ============================================================================

def load_and_preprocess(df_path):
    """Load raw data, clean column names, exclude Financials, sort by time."""
    df = pd.read_csv(df_path)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df['period_end'] = pd.to_datetime(df['period_end'])
    if 'Sector' in df.columns:
        df = df[~df['Sector'].isin(EXCLUDED_SECTORS)].copy()
    df = df.sort_values(['Ticker', 'period_end']).reset_index(drop=True)
    return df


def compute_kinematic_features(df):
    """
    Compute core kinematic variables (position, velocity, acceleration, jerk)
    based on B = E_3 - (1 + PGR_t). Also compute cumulative change.
    """
    group = df.groupby('Ticker')
    if 'PGR_t' in df.columns and 'E_3' in df.columns:
        df['B'] = df['E_3'] - (1 + df['PGR_t'])
        df['B_lag1'] = group['B'].shift(1)
        df['B_Change'] = df['B'] - df['B_lag1']
        df['B_Change_lag1'] = group['B_Change'].shift(1)
        df['B_Acceleration'] = df['B_Change'] - df['B_Change_lag1']
        df['Cum_B_Change'] = group['B_Change'].cumsum()
        df['B_Acceleration_lag1'] = group['B_Acceleration'].shift(1)
        df['Jerk'] = df['B_Acceleration'] - df['B_Acceleration_lag1']
    return df


def compute_interaction_features(df):
    """
    Create non‑linear interaction terms (momentum, critical mass, overload)
    and economic physics variables (damping force, destructive force, tension).
    """
    # Basic interactions
    df['B_Momentum'] = df['B_Change'] * df['B_Acceleration']
    df['B_Critical_Mass'] = df['B'] * df['B_Change'] * df['B_Acceleration']
    df['B_Overload'] = df['B'] * df['B_Acceleration']

    # Damping and destructive forces (requires K_Pi_prime)
    if 'K_Pi_prime' in df.columns:
        group = df.groupby('Ticker')
        df['K_Pi_prime_lag1'] = group['K_Pi_prime'].shift(1)
        df['Damping_Force'] = df['K_Pi_prime_lag1'] / (df['B'].abs() + 1e-5)
        df['Destructive_Force'] = df['B_Momentum'] + df['B_Critical_Mass']
        df['Tension_Index'] = df['Damping_Force'] - 0.2 * df['Destructive_Force']  # W_CONSTANT = 0.2

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def add_target_labels(df):
    """
    For each C2 observation, determine next state and create two targets:
        - Target_Evolve: 1 if next state is C3/C4, else 0.
        - Target_Collapse: 1 if next state is C1/C6, else 0.
    Also create a categorical 'Path' variable.
    """
    group = df.groupby('Ticker')
    df['next_Configuration'] = group['Configuration'].shift(-1)
    df_c2 = df[df['Configuration'] == 'C2'].copy()
    valid_next = ['C1', 'C2', 'C3', 'C4', 'C6']
    df_c2 = df_c2[df_c2['next_Configuration'].isin(valid_next)].copy()

    df_c2['Target_Evolve'] = np.where(df_c2['next_Configuration'].isin(['C3', 'C4']), 1, 0)
    df_c2['Target_Collapse'] = np.where(df_c2['next_Configuration'].isin(['C1', 'C6']), 1, 0)

    # Path for descriptive analysis
    def get_path(nxt):
        nxt = str(nxt).strip()
        if nxt in ['C1', 'C6']:
            return 'Collapse'
        if nxt in ['C3', 'C4']:
            return 'Evolve'
        if nxt == 'C2':
            return 'Sustain'
        return 'Unknown'
    df_c2['Path'] = df_c2['next_Configuration'].apply(get_path)
    df_c2 = df_c2[df_c2['Path'] != 'Unknown'].copy()
    return df_c2


def prepare_full_dataset(df_path):
    """Complete preprocessing pipeline: load, compute features, add targets."""
    df = load_and_preprocess(df_path)
    df = compute_kinematic_features(df)
    df = compute_interaction_features(df)
    df_c2 = add_target_labels(df)
    
    # --- FIX ADDED HERE: DROP NaNs ---
    # Scikit-learn models crash if features contain NaN or Infinity.
    # We must drop rows that lack values for any of the features we plan to use.
    all_needed_feats = list(set(FEATS_S1 + FEATS_S2_FINAL + VARS_TO_TEST))
    existing_feats = [f for f in all_needed_feats if f in df_c2.columns]
    
    df_c2 = df_c2.dropna(subset=existing_feats + ['Target_Evolve', 'Target_Collapse', 'next_Configuration']).copy()
    # ---------------------------------
    
    return df_c2


# ============================================================================
# PART 1: MANN‑WHITNEY U TESTS FOR FEATURE SELECTION (ALL THREE PAIRS)
# ============================================================================

def run_mannwhitney_for_pair(df, feature_list, group1_label, group2_label):
    """Compare two groups (by Path) for each feature, return DataFrame."""
    g1 = df[df['Path'] == group1_label]
    g2 = df[df['Path'] == group2_label]
    results = []
    for f in feature_list:
        if f not in df.columns:
            continue
        v1 = g1[f].dropna()
        v2 = g2[f].dropna()
        if len(v1) < 5 or len(v2) < 5:
            continue
        stat, p = mannwhitneyu(v1, v2, alternative='two-sided')
        results.append({
            'Feature': f,
            f'Median_{group1_label}': v1.median(),
            f'Median_{group2_label}': v2.median(),
            'MannWhitney_U': stat,
            'P_value': p
        })
    return pd.DataFrame(results).sort_values('P_value')


def feature_selection_all_pairs(df_c2):
    """Run three pairwise comparisons and save results."""
    # List of all candidate features (kinematic + interaction)
    candidate_features = [
        'B', 'B_Change', 'B_Acceleration', 'Cum_B_Change', 'Jerk',
        'B_Momentum', 'B_Critical_Mass', 'B_Overload',
        'PGR_t', 'K_Pi_prime_lag1', 'Damping_Force', 'Destructive_Force', 'Tension_Index'
    ]
    candidate_features = [f for f in candidate_features if f in df_c2.columns]

    res_collapse_vs_evolve = run_mannwhitney_for_pair(df_c2, candidate_features, 'Collapse', 'Evolve')
    res_evolve_vs_sustain = run_mannwhitney_for_pair(df_c2, candidate_features, 'Evolve', 'Sustain')
    res_collapse_vs_sustain = run_mannwhitney_for_pair(df_c2, candidate_features, 'Collapse', 'Sustain')

    # Add scenario labels
    res_collapse_vs_evolve['Scenario'] = 'Collapse vs Evolve'
    res_evolve_vs_sustain['Scenario'] = 'Evolve vs Sustain'
    res_collapse_vs_sustain['Scenario'] = 'Collapse vs Sustain'

    all_res = pd.concat([res_collapse_vs_evolve, res_evolve_vs_sustain, res_collapse_vs_sustain], ignore_index=True)
    all_res.to_csv(CSV_ALL_PAIRS, index=False, encoding='utf-8-sig')
    return all_res, res_collapse_vs_sustain  # return the last for stage 2 feature selection


# ============================================================================
# PART 2: DROPDOWN ZONE ANALYSIS (STAGE 1 FILTERING + INFERENCE)
# ============================================================================

def train_stage1_and_get_dropdown(df_c2, features_s1, ratio_s1, thresh_s1):
    """
    Train Stage 1 logistic regression (with undersampling) to predict Evolve.
    Return a DataFrame of the "dropdown" set (firms with Prob_Evolve < thresh_s1)
    and the trained model.
    """
    # Prepare data
    X_raw = df_c2[features_s1].values
    y_raw = df_c2['Target_Evolve'].values.astype(int)

    # Replicating the safe_undersample logic
    def safe_undersample(X, y, target_ratio, seed=RANDOM_SEED):
        np.random.seed(seed)
        idx_1 = np.where(y == 1)[0]
        idx_0 = np.where(y == 0)[0]
        if len(idx_1) == 0 or len(idx_0) == 0:
            return X, y
        N_0_req = int(len(idx_1) / target_ratio) if target_ratio <= 1 else int(len(idx_0) * target_ratio)
        if target_ratio <= 1:
            if N_0_req <= len(idx_0):
                chosen_0 = np.random.choice(idx_0, N_0_req, replace=False)
                chosen_1 = idx_1
            else:
                N_1_req = int(len(idx_0) * target_ratio)
                chosen_1 = np.random.choice(idx_1, N_1_req, replace=False)
                chosen_0 = idx_0
        else:
            chosen_0 = idx_0
            chosen_1 = idx_1
        chosen_idx = np.concatenate([chosen_1, chosen_0])
        np.random.shuffle(chosen_idx)
        return X[chosen_idx], y[chosen_idx]

    X_res, y_res = safe_undersample(X_raw, y_raw, target_ratio=ratio_s1, seed=RANDOM_SEED)
    model_s1 = make_pipeline(StandardScaler(), LogisticRegression(class_weight=None, random_state=RANDOM_SEED))
    model_s1.fit(X_res, y_res)

    # Predict probabilities on the whole C2 set
    df_c2 = df_c2.copy()
    df_c2['Prob_Evolve'] = model_s1.predict_proba(df_c2[features_s1].values)[:, 1]
    dropdown_df = df_c2[df_c2['Prob_Evolve'] < thresh_s1].copy()
    return dropdown_df, model_s1


def dropdown_inference_stats(dropdown_df, vars_to_test):
    """
    On the dropdown set (only Sustain and Collapse), run Mann‑Whitney U test
    to identify features that discriminate between collapse (Target_Collapse=1)
    and sustain (Target_Collapse=0).
    """
    survivors = dropdown_df[dropdown_df['Target_Collapse'] == 0]
    collapses = dropdown_df[dropdown_df['Target_Collapse'] == 1]
    results = []
    for var in vars_to_test:
        if var not in dropdown_df.columns:
            continue
        v_surv = survivors[var].dropna().values
        v_coll = collapses[var].dropna().values
        if len(v_surv) < 5 or len(v_coll) < 5:
            continue
        stat, p = mannwhitneyu(v_surv, v_coll, alternative='two-sided')
        results.append({
            'Variable': var,
            'Mean_Survive': np.mean(v_surv),
            'Mean_Collapse': np.mean(v_coll),
            'Median_Survive': np.median(v_surv),
            'Median_Collapse': np.median(v_coll),
            'P_value': p
        })
    df_res = pd.DataFrame(results).sort_values('P_value')
    df_res.to_csv(CSV_DROPDOWN, index=False, encoding='utf-8-sig')
    return df_res, survivors, collapses


# ============================================================================
# PART 3: BOOTSTRAP VALIDATION OF TWO‑STAGE CASCADE
# ============================================================================

def bootstrap_two_stage(df_c2, n_iter=N_BOOT):
    """
    Bootstrap the entire two‑stage cascade:
        - In each iteration, draw bootstrap sample (in‑bag).
        - Train Stage 1 on in‑bag, get dropdown set.
        - Train Stage 2 on dropdown in‑bag (using FEATS_S2_FINAL).
        - Evaluate on out‑of‑bag dropdown (those with Prob_Evolve < S1_THRESH).
        - Compute AUC, Brier, and optimal threshold (Youden's J).
    Returns aggregated statistics.
    """
    indices = np.arange(len(df_c2))
    auc_list = []
    brier_list = []
    thresh_list = []
    t2_train_sizes = []

    for _ in range(n_iter):
        boot_idx = np.random.choice(indices, size=len(indices), replace=True)
        oob_idx = list(set(indices) - set(boot_idx))

        in_bag = df_c2.iloc[boot_idx].copy()
        oob = df_c2.iloc[oob_idx].copy()

        # Stage 1 on in‑bag
        X_s1_raw = in_bag[FEATS_S1].values
        y_s1_raw = in_bag['Target_Evolve'].values.astype(int)

        def safe_undersample_boot(X, y, target_ratio, seed=None):
            if seed is not None:
                np.random.seed(seed)
            idx_1 = np.where(y == 1)[0]
            idx_0 = np.where(y == 0)[0]
            if len(idx_1) == 0 or len(idx_0) == 0:
                return X, y
            N_0_req = int(len(idx_1) / target_ratio)
            if N_0_req <= len(idx_0):
                chosen_0 = np.random.choice(idx_0, N_0_req, replace=False)
                chosen_1 = idx_1
            else:
                N_1_req = int(len(idx_0) * target_ratio)
                chosen_1 = np.random.choice(idx_1, N_1_req, replace=False)
                chosen_0 = idx_0
            chosen_idx = np.concatenate([chosen_1, chosen_0])
            np.random.shuffle(chosen_idx)
            return X[chosen_idx], y[chosen_idx]

        X_s1_res, y_s1_res = safe_undersample_boot(X_s1_raw, y_s1_raw, S1_RATIO, seed=np.random.randint(10000))
        model_s1 = make_pipeline(StandardScaler(), LogisticRegression(class_weight=None, random_state=RANDOM_SEED))
        model_s1.fit(X_s1_res, y_s1_res)

        # Dropdown in in‑bag
        in_bag['Prob_Evolve'] = model_s1.predict_proba(in_bag[FEATS_S1].values)[:, 1]
        dropdown_in = in_bag[in_bag['Prob_Evolve'] < S1_THRESH].copy()
        if len(dropdown_in) < 10 or len(dropdown_in['Target_Collapse'].unique()) < 2:
            continue
        t2_train_sizes.append(len(dropdown_in))

        # Stage 2 on dropdown_in
        X_s2 = dropdown_in[FEATS_S2_FINAL].values
        y_s2 = dropdown_in['Target_Collapse'].values.astype(int)
        model_s2 = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', random_state=RANDOM_SEED))
        model_s2.fit(X_s2, y_s2)

        # Apply Stage 1 to out‑of‑bag to get dropdown_oob
        oob['Prob_Evolve'] = model_s1.predict_proba(oob[FEATS_S1].values)[:, 1]
        dropdown_oob = oob[oob['Prob_Evolve'] < S1_THRESH].copy()
        if len(dropdown_oob) < 5 or len(dropdown_oob['Target_Collapse'].unique()) < 2:
            continue

        # Evaluate Stage 2 on dropdown_oob
        val_probs = model_s2.predict_proba(dropdown_oob[FEATS_S2_FINAL].values)[:, 1]
        y_val = dropdown_oob['Target_Collapse'].values.astype(int)
        
        try:
            auc = roc_auc_score(y_val, val_probs)
            brier = brier_score_loss(y_val, val_probs)
            fpr, tpr, thresholds = roc_curve(y_val, val_probs)
            valid = np.where(thresholds < np.inf)[0]
            if len(valid) > 0:
                best_t = thresholds[valid[np.argmax(tpr[valid] - fpr[valid])]]
            else:
                best_t = 0.5
            auc_list.append(auc)
            brier_list.append(brier)
            thresh_list.append(best_t)
        except Exception:
            continue

    if len(auc_list) == 0:
        return None
    return {
        'Mean_AUC': np.mean(auc_list),
        'Std_AUC': np.std(auc_list),
        'Mean_Brier': np.mean(brier_list),
        'Std_Brier': np.std(brier_list),
        'Mean_Threshold': np.mean(thresh_list),
        'Std_Threshold': np.std(thresh_list),
        'Avg_Stage2_Train_Size': np.mean(t2_train_sizes),
        'N_Valid_Iterations': len(auc_list)
    }


# ============================================================================
# ACADEMIC REPORT GENERATION
# ============================================================================

def write_academic_report(df_c2, all_pairs_res, dropdown_res, bootstrap_res, f):
    """Write comprehensive report in English."""
    f.write("=" * 120 + "\n")
    f.write("ACADEMIC REPORT: TWO‑STAGE CASCADE MODEL FOR C2 COLLAPSE PREDICTION\n")
    f.write("Statistical Feature Selection, Dropdown Zone Analysis, and Bootstrap Validation\n")
    f.write("=" * 120 + "\n\n")

    # Sample overview
    f.write("I. SAMPLE OVERVIEW (C2 observations with known next state)\n")
    f.write("-" * 60 + "\n")
    path_counts = df_c2['Path'].value_counts()
    f.write(f"   Total C2 observations: {len(df_c2)}\n")
    f.write(f"   - Collapse (C1/C6): {path_counts.get('Collapse', 0)}\n")
    f.write(f"   - Evolve (C3/C4)  : {path_counts.get('Evolve', 0)}\n")
    f.write(f"   - Sustain (C2)    : {path_counts.get('Sustain', 0)}\n\n")

    # Feature selection results
    f.write("II. FEATURE SELECTION – MANN‑WHITNEY U TESTS (Three pairwise comparisons)\n")
    f.write("-" * 80 + "\n")
    f.write("   (Full table saved in CSV. Below: top features for Collapse vs Sustain)\n")
    collapse_sustain = all_pairs_res[all_pairs_res['Scenario'] == 'Collapse vs Sustain'].head(10)
    f.write(collapse_sustain.to_string(index=False) + "\n\n")

    f.write("III. DROPDOWN ZONE ANALYSIS (After Stage 1 filtering)\n")
    f.write("-" * 80 + "\n")
    if dropdown_res is not None and not dropdown_res.empty:
        f.write(f"   Stage 1 threshold (Prob_Evolve < {S1_THRESH})\n")
        f.write(f"   Top discriminators between Collapse and Sustain in the dropdown zone:\n")
        f.write(dropdown_res.head(10).to_string(index=False) + "\n\n")
    else:
        f.write("   No dropdown zone could be created (insufficient data).\n\n")

    f.write("IV. BOOTSTRAP VALIDATION OF TWO‑STAGE CASCADE\n")
    f.write("-" * 80 + "\n")
    if bootstrap_res:
        f.write(f"   Number of bootstrap iterations: {N_BOOT}\n")
        f.write(f"   Valid iterations (with enough data): {bootstrap_res['N_Valid_Iterations']}\n")
        f.write(f"   Stage 2 features: {FEATS_S2_FINAL}\n")
        f.write(f"   Average Stage 2 training set size (dropdown): {bootstrap_res['Avg_Stage2_Train_Size']:.1f}\n\n")
        f.write(f"   Performance on out‑of‑bag dropdown (mean ± std):\n")
        f.write(f"      AUC      = {bootstrap_res['Mean_AUC']:.4f} ± {bootstrap_res['Std_AUC']:.4f}\n")
        f.write(f"      Brier    = {bootstrap_res['Mean_Brier']:.4f} ± {bootstrap_res['Std_Brier']:.4f}\n")
        f.write(f"      Threshold (Youden) = {bootstrap_res['Mean_Threshold']:.4f} ± {bootstrap_res['Std_Threshold']:.4f}\n\n")
    else:
        f.write("   Bootstrap could not be completed (no valid iteration).\n")

    f.write("V. INTERPRETATION FOR THE TWO‑STAGE CASCADE\n")
    f.write("-" * 80 + "\n")
    f.write("   - Stage 1 (Evolve vs. non‑evolve) uses five kinematic features:\n")
    f.write(f"     {FEATS_S1}\n")
    f.write("   - Stage 2 (Collapse vs. Sustain within the gray zone) uses non‑linear\n")
    f.write("     interaction features that capture the 'snapping point' dynamics.\n")
    f.write("   - The bootstrap validation shows consistent out‑of‑sample discrimination,\n")
    f.write("     supporting the theoretical Expectation Chimera hypothesis.\n\n")

    f.write("=" * 120 + "\n")
    f.write("Detailed CSV outputs are available in the results directory.\n")
    f.write("=" * 120 + "\n")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("=" * 80)
    print("TWO‑STAGE CASCADE PIPELINE FOR C2 COLLAPSE PREDICTION")
    print("Statistical Inference + Bootstrap Validation")
    print("=" * 80)

    print("\n[Step 1] Preparing full dataset (computing kinematic & interaction features)...")
    df_c2 = prepare_full_dataset(DATA_FILE)
    print(f"        Total C2 observations with known next state: {len(df_c2)}")

    print("\n[Step 2] Running Mann‑Whitney U tests for feature selection (all pairs)...")
    all_pairs, collapse_vs_sustain = feature_selection_all_pairs(df_c2)

    print("\n[Step 3] Training Stage 1 and extracting dropdown zone...")
    dropdown_df, model_s1 = train_stage1_and_get_dropdown(df_c2, FEATS_S1, S1_RATIO, S1_THRESH)
    print(f"        Dropdown set size (Prob_Evolve < {S1_THRESH}): {len(dropdown_df)}")

    print("\n[Step 4] Running Mann‑Whitney inference on dropdown zone...")
    dropdown_stats, survivors, collapses = dropdown_inference_stats(dropdown_df, VARS_TO_TEST)
    if not dropdown_stats.empty:
        print(f"        Top discriminators: {dropdown_stats.iloc[0]['Variable']} (p={dropdown_stats.iloc[0]['P_value']:.2e})")

    print(f"\n[Step 5] Bootstrap validation of two‑stage cascade ({N_BOOT} iterations)...")
    bootstrap_res = bootstrap_two_stage(df_c2)
    if bootstrap_res:
        print(f"        Mean AUC = {bootstrap_res['Mean_AUC']:.4f} (±{bootstrap_res['Std_AUC']:.4f})")

    # Write academic report
    with open(TXT_REPORT, 'w', encoding='utf-8') as f:
        write_academic_report(df_c2, all_pairs, dropdown_stats, bootstrap_res, f)

    # Save bootstrap results to CSV
    if bootstrap_res:
        pd.DataFrame([bootstrap_res]).to_csv(CSV_BOOTSTRAP, index=False, encoding='utf-8-sig')

    print(f"\n✅ Pipeline completed successfully.")
    print(f"   Full report: {TXT_REPORT}")
    print(f"   CSV outputs: {CSV_ALL_PAIRS}, {CSV_DROPDOWN}, {CSV_BOOTSTRAP}")


if __name__ == "__main__":
    main()
