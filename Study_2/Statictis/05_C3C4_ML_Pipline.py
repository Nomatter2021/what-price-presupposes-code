"""
05_C3C4_ML_Pipeline.py

Academic Title:
    Machine Learning Pipeline for Predicting Collapse from C3/C4 States:
    Statistical Feature Selection, Grid Search, and Bootstrap Robustness

Objectives:
    This script implements a complete machine learning workflow to predict
    whether a firm in state C3 or C4 will transition into a crash (C1/C6)
    in the next quarter. The analysis excludes the Financials sector.

Steps:
    1. Statistical Feature Selection (Mann‑Whitney U Test):
       - Features that do NOT differ between C3 and C4 (p > 0.05) – “Stay” signal.
       - Features that DO differ between C3/C4 and crash states (p < 0.05) – “Collapse” signal.
       - Features satisfying both criteria are considered “Candidates”.

    2. Grid Search (Correct Timeframe):
       - Use only past and current information (t, t-1, t-2) to predict collapse at t+1.
       - Logistic regression with standard scaling and random undersampling.
       - Optimise threshold on training set (Youden’s J) and evaluate on test set.
       - Evaluate different feature combinations, resampling ratios, and test splits.
       - Ranking by Recall → F1 → AUC.

    3. Bootstrap Robustness (200 iterations):
       - For the two most promising combinations (E and F).
       - Compute 95% confidence intervals for Recall, F1, AUC.
       - Assess stability of performance.

Output Files (saved in data/results/):
    - 04_ML_Pipeline_report.txt         : Full academic report.
    - 04_statistical_filter_results.csv : Mann‑Whitney p‑values and candidate flags.
    - 04_grid_search_results.csv        : Grid search results with ranking.
    - 04_robustness_results.csv         : Bootstrap statistics for top combinations.

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
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, roc_curve
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_FILE = Path('../data/final_panel.csv')
OUTPUT_DIR = Path('results')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TXT_REPORT = OUTPUT_DIR / 'report/05_ML_Pipeline_report.txt'
CSV_FILTER = OUTPUT_DIR / 'table/05_statistical_filter_results.csv'
CSV_GRID = OUTPUT_DIR / 'table/05_grid_search_results.csv'
CSV_ROBUST = OUTPUT_DIR / 'table/05_robustness_results.csv'

RANDOM_SEED = 42
N_BOOT = 200
TEST_SPLITS = [0.125, 0.2]
RESAMPLING_RATIOS = {'1:1': 1.0, '1.5:1': 0.666, '2:1': 0.5, '3:1': 0.333, '4:1': 0.25, '5:1': 0.2}
GRID_COMBINATIONS = {
    'A. PDI_t (current governance)': ['PDI_t'],
    'B. PDI_roll_mean (governance trend)': ['PDI_roll_mean'],
    'C. PDI_t + K_Pi_prime_t (current comprehensive)': ['PDI_t', 'K_Pi_prime_t'],
    'D. PDI_t + K_Pi_prime_lag1 (current governance + past capital)': ['PDI_t', 'K_Pi_prime_lag1'],
    'E. PDI_roll_mean + K_Pi_prime_t (governance trend + current capital)': ['PDI_roll_mean', 'K_Pi_prime_t'],
    'F. PDI_roll_mean + K_Pi_prime_lag1 (governance trend + past capital)': ['PDI_roll_mean', 'K_Pi_prime_lag1']
}
ROBUST_COMBOS = {
    'E. PDI_roll_mean + K_Pi_prime_t': ['PDI_roll_mean', 'K_Pi_prime_t'],
    'F. PDI_roll_mean + K_Pi_prime_lag1': ['PDI_roll_mean', 'K_Pi_prime_lag1']
}
np.random.seed(RANDOM_SEED)


# ============================================================================
# DATA PREPARATION FUNCTIONS
# ============================================================================

def load_and_filter_data(exclude_financials=True):
    """Load raw data, apply formal gate, and optionally exclude Financials sector."""
    df = pd.read_csv(DATA_FILE)
    df['period_end'] = pd.to_datetime(df['period_end'])
    if 'Regime_Label' in df.columns:
        df['Configuration'] = np.where(df['Regime_Label'] == 'Normal_Regime', 'Normal', df['Configuration'])
    if exclude_financials and 'Sector' in df.columns:
        financial_labels = ['Financials_and_Real_Estate', 'Financial']
        df = df[~df['Sector'].isin(financial_labels)].copy()
    return df


def preprocess_for_statistical_filter(df):
    """Prepare data for Mann‑Whitney U test (C3/C4 phase, next state known)."""
    df = df.sort_values(['Ticker', 'period_end']).reset_index(drop=True)
    group_col = 'Ticker'
    df['next_Configuration'] = df.groupby(group_col)['Configuration'].shift(-1)

    # Derived features
    if 'PGR_t' in df.columns and 'E_3' in df.columns:
        df['B'] = df['E_3'] - (1 + df['PGR_t'])
    else:
        df['B'] = np.nan

    df['PDI_lag1'] = df.groupby(group_col)['PDI_t'].shift(1)
    df['PDI_change'] = df['PDI_t'] - df['PDI_lag1']
    df['PDI_roll_mean'] = df.groupby(group_col)['PDI_t'].transform(
        lambda x: x.rolling(3, min_periods=2).mean()
    )

    if 'K_Pi_prime' in df.columns:
        df['K_Pi_prime_lag1'] = df.groupby(group_col)['K_Pi_prime'].shift(1)
        df['dK_Pi_prime'] = df['K_Pi_prime'] - df['K_Pi_prime_lag1']

    # Keep only rows in C3 or C4 with known next state
    df_c3c4 = df[df['Configuration'].isin(['C3', 'C4'])].copy()
    df_c3c4 = df_c3c4.dropna(subset=['next_Configuration'])
    return df_c3c4


def preprocess_for_grid_and_robustness(df):
    """Prepare data for prediction: use only past/current info to predict t+1."""
    df = df.sort_values(['Ticker', 'period_end']).reset_index(drop=True)
    group_col = 'Ticker'
    df['next_Configuration'] = df.groupby(group_col)['Configuration'].shift(-1)

    # Features: current and lagged (only information available at time t)
    df['PDI_lag1'] = df.groupby(group_col)['PDI_t'].shift(1)
    df['PDI_roll_mean'] = df.groupby(group_col)['PDI_t'].transform(
        lambda x: x.rolling(3, min_periods=2).mean()
    )

    if 'K_Pi_prime' in df.columns:
        df['K_Pi_prime_t'] = df['K_Pi_prime']
        df['K_Pi_prime_lag1'] = df.groupby(group_col)['K_Pi_prime'].shift(1)

    df_c3c4 = df[df['Configuration'].isin(['C3', 'C4'])].copy()
    df_c3c4['collapse_target'] = np.where(
        df_c3c4['next_Configuration'].isin(['C1', 'C6']), 1,
        np.where(df_c3c4['next_Configuration'].isin(['C3', 'C4']), 0, np.nan)
    )
    df_c3c4 = df_c3c4.dropna(subset=['collapse_target']).copy()

    # Fill NaNs for lagged features with 0 (conservative assumption)
    fill_cols = ['PDI_lag1', 'PDI_roll_mean', 'K_Pi_prime_t', 'K_Pi_prime_lag1']
    for col in fill_cols:
        if col in df_c3c4.columns:
            df_c3c4[col] = df_c3c4[col].fillna(0)
    return df_c3c4


# ============================================================================
# PART 1: STATISTICAL FEATURE SELECTION (MANN‑WHITNEY U)
# ============================================================================

def run_statistical_filter(df_c3c4):
    """Perform Mann‑Whitney U tests for each feature and return results DataFrame."""
    features = [
        'PDI_t', 'PDI_lag1', 'PDI_change', 'PDI_roll_mean',
        'R_t', 'E_3', 'B', 'PGR_t',
        'K_Pi_prime', 'K_Pi_prime_lag1', 'dK_Pi_prime'
    ]

    # Masks
    c3_mask = (df_c3c4['Configuration'] == 'C3') & (df_c3c4['next_Configuration'].isin(['C3', 'C4']))
    c4_mask = (df_c3c4['Configuration'] == 'C4') & (df_c3c4['next_Configuration'].isin(['C3', 'C4']))
    col_c3_mask = (df_c3c4['Configuration'] == 'C3') & (df_c3c4['next_Configuration'].isin(['C1', 'C6']))
    col_c4_mask = (df_c3c4['Configuration'] == 'C4') & (df_c3c4['next_Configuration'].isin(['C1', 'C6']))
    stay_mask = df_c3c4['next_Configuration'].isin(['C3', 'C4'])
    col_mask = df_c3c4['next_Configuration'].isin(['C1', 'C6'])

    results = []
    for feat in features:
        if feat not in df_c3c4.columns:
            continue

        c3_vals = df_c3c4.loc[c3_mask, feat].dropna().values
        c4_vals = df_c3c4.loc[c4_mask, feat].dropna().values
        col_c3_vals = df_c3c4.loc[col_c3_mask, feat].dropna().values
        col_c4_vals = df_c3c4.loc[col_c4_mask, feat].dropna().values
        stay_vals = df_c3c4.loc[stay_mask, feat].dropna().values
        col_vals = df_c3c4.loc[col_mask, feat].dropna().values

        p_c3_c4 = mannwhitneyu(c3_vals, c4_vals, alternative='two-sided')[1] if len(c3_vals) and len(c4_vals) else np.nan
        p_c3_col = mannwhitneyu(c3_vals, col_c3_vals, alternative='two-sided')[1] if len(c3_vals) and len(col_c3_vals) else np.nan
        p_c4_col = mannwhitneyu(c4_vals, col_c4_vals, alternative='two-sided')[1] if len(c4_vals) and len(col_c4_vals) else np.nan
        p_stay_col = mannwhitneyu(stay_vals, col_vals, alternative='two-sided')[1] if len(stay_vals) and len(col_vals) else np.nan

        pass_c3_c4 = p_c3_c4 > 0.05 if not np.isnan(p_c3_c4) else False
        pass_c3_col = p_c3_col < 0.05 if not np.isnan(p_c3_col) else False
        pass_c4_col = p_c4_col < 0.05 if not np.isnan(p_c4_col) else False
        is_candidate = pass_c3_c4 and (pass_c3_col or pass_c4_col)

        results.append({
            'Feature': feat,
            'p_C3_vs_C4': p_c3_c4,
            'Pass_C3_C4 (Stay)': pass_c3_c4,
            'p_C3_vs_Collapse': p_c3_col,
            'Pass_C3_Collapse': pass_c3_col,
            'p_C4_vs_Collapse': p_c4_col,
            'Pass_C4_Collapse': pass_c4_col,
            'p_Stay_vs_Collapse': p_stay_col,
            'Is_Candidate': is_candidate
        })

    return pd.DataFrame(results)


# ============================================================================
# PART 2: GRID SEARCH (CORRECT TIMEFRAME)
# ============================================================================

def evaluate_config(X_train, y_train, X_test, y_test, sampling_ratio):
    """Train logistic regression with undersampling, tune threshold on train, evaluate on test."""
    rus = RandomUnderSampler(sampling_strategy=sampling_ratio, random_state=RANDOM_SEED)
    try:
        X_res, y_res = rus.fit_resample(X_train, y_train)
    except ValueError:
        X_res, y_res = X_train, y_train

    model = make_pipeline(StandardScaler(), LogisticRegression(class_weight=None, random_state=RANDOM_SEED))
    model.fit(X_res, y_res)

    # Optimal threshold on training set (Youden's J)
    y_train_proba = model.predict_proba(X_res)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_res, y_train_proba)
    best_thresh = thresholds[np.argmax(tpr - fpr)] if len(thresholds) > 0 else 0.5

    # Test
    y_test_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_test_proba >= best_thresh).astype(int)
    auc = roc_auc_score(y_test, y_test_proba)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
    return auc, prec, rec, f1


def run_grid_search(df_ml, combinations, test_sizes, ratios):
    """Grid search over feature combos, test splits, and resampling ratios."""
    results = []
    for test_size in test_sizes:
        for ratio_name, ratio_val in ratios.items():
            for combo_name, features in combinations.items():
                missing = [f for f in features if f not in df_ml.columns]
                if missing:
                    continue
                temp = df_ml.dropna(subset=features + ['collapse_target'])
                X = temp[features].values
                y = temp['collapse_target'].values
                if len(np.unique(y)) < 2:
                    continue

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, stratify=y, random_state=RANDOM_SEED
                )
                auc, prec, rec, f1 = evaluate_config(X_train, y_train, X_test, y_test, ratio_val)
                results.append({
                    'Test_Size': test_size,
                    'Resampling_Ratio': ratio_name,
                    'Combination': combo_name,
                    'Recall': rec,
                    'F1': f1,
                    'AUC': auc,
                    'Precision': prec
                })
    df_results = pd.DataFrame(results)
    # Sort by Recall -> F1 -> AUC
    df_results = df_results.sort_values(by=['Recall', 'F1', 'AUC'], ascending=[False, False, False])
    return df_results


# ============================================================================
# PART 3: BOOTSTRAP ROBUSTNESS (TOP COMBOS)
# ============================================================================

def bootstrap_evaluate(X_all, y_all, sampling_ratio, n_iter=N_BOOT):
    """Bootstrap with train/test split (87.5% / 12.5%) and compute performance metrics."""
    results = []
    for _ in range(n_iter):
        idx = np.random.choice(len(X_all), len(X_all), replace=True)
        X_boot, y_boot = X_all[idx], y_all[idx]
        split = int(0.875 * len(X_boot))
        X_train, X_test = X_boot[:split], X_boot[split:]
        y_train, y_test = y_boot[:split], y_boot[split:]
        if len(np.unique(y_train)) > 1 and len(np.unique(y_test)) > 1:
            auc, prec, rec, f1 = evaluate_config(X_train, y_train, X_test, y_test, sampling_ratio)
            results.append({'AUC': auc, 'Recall': rec, 'F1': f1})
    return pd.DataFrame(results)


def run_robustness_analysis(df_ml, robust_combos, ratios):
    """For each combo and resampling ratio, bootstrap and compute CI."""
    summary = []
    for combo_name, features in robust_combos.items():
        missing = [f for f in features if f not in df_ml.columns]
        if missing:
            continue
        X_all = df_ml[features].values
        y_all = df_ml['collapse_target'].values
        for ratio_name, ratio_val in ratios.items():
            boot_df = bootstrap_evaluate(X_all, y_all, ratio_val)
            if boot_df.empty:
                continue
            summary.append({
                'Combination': combo_name,
                'Resampling_Ratio': ratio_name,
                'Recall_mean': boot_df['Recall'].mean(),
                'Recall_CI_low': np.percentile(boot_df['Recall'], 2.5),
                'Recall_CI_high': np.percentile(boot_df['Recall'], 97.5),
                'F1_mean': boot_df['F1'].mean(),
                'F1_CI_low': np.percentile(boot_df['F1'], 2.5),
                'F1_CI_high': np.percentile(boot_df['F1'], 97.5),
                'AUC_mean': boot_df['AUC'].mean(),
                'AUC_CI_low': np.percentile(boot_df['AUC'], 2.5),
                'AUC_CI_high': np.percentile(boot_df['AUC'], 97.5)
            })
    return pd.DataFrame(summary)


# ============================================================================
# REPORT GENERATION
# ============================================================================

def write_academic_report(filter_df, grid_df, robust_df, f):
    """Write comprehensive academic report to file."""
    f.write("=" * 110 + "\n")
    f.write("MACHINE LEARNING PIPELINE FOR COLLAPSE PREDICTION FROM C3/C4 STATES\n")
    f.write("Statistical Feature Selection, Grid Search, and Bootstrap Robustness\n")
    f.write("=" * 110 + "\n\n")

    # Part 1: Statistical Filter
    f.write("1. STATISTICAL FEATURE SELECTION (Mann‑Whitney U Test)\n")
    f.write("-" * 60 + "\n")
    f.write("Criteria:\n")
    f.write("  - Stay signal   : No difference between C3 and C4 (p > 0.05)\n")
    f.write("  - Collapse signal: Difference between C3/C4 and C1/C6 (p < 0.05)\n")
    f.write("  - Candidate features satisfy both.\n\n")
    candidate_feats = filter_df[filter_df['Is_Candidate']]['Feature'].tolist()
    if candidate_feats:
        f.write(f"Candidate features: {', '.join(candidate_feats)}\n\n")
    else:
        f.write("No candidate features found.\n\n")
    f.write("Detailed results (first 10 rows):\n")
    f.write(filter_df.head(10).to_string(index=False) + "\n\n")

    # Part 2: Grid Search
    f.write("2. GRID SEARCH (Correct Timeframe: past/current info → t+1)\n")
    f.write("-" * 60 + "\n")
    f.write("Top 15 configurations ranked by Recall → F1 → AUC:\n")
    f.write(grid_df.head(15).to_string(index=False) + "\n\n")

    # Part 3: Bootstrap Robustness
    f.write("3. BOOTSTRAP ROBUSTNESS (200 iterations, 95% CI)\n")
    f.write("-" * 60 + "\n")
    f.write("For the two most promising combinations (E and F) and all resampling ratios:\n")
    f.write(robust_df.to_string(index=False) + "\n\n")
    f.write("=" * 110 + "\n")
    f.write("CSV files with full results are saved in the output directory.\n")
    f.write("=" * 110 + "\n")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("=" * 80)
    print("MACHINE LEARNING PIPELINE: COLLAPSE PREDICTION FROM C3/C4")
    print("Step 1: Statistical Feature Selection")
    print("Step 2: Grid Search")
    print("Step 3: Bootstrap Robustness")
    print("=" * 80)

    # Load base data
    df_raw = load_and_filter_data(exclude_financials=True)

    # ----- PART 1: Statistical Filter -----
    print("\n[Step 1] Running Mann‑Whitney U tests...")
    df_filter = preprocess_for_statistical_filter(df_raw)
    filter_results = run_statistical_filter(df_filter)
    filter_results.to_csv(CSV_FILTER, index=False, encoding='utf-8-sig')

    # ----- PART 2: Grid Search -----
    print("[Step 2] Running grid search...")
    df_ml = preprocess_for_grid_and_robustness(df_raw)
    grid_results = run_grid_search(df_ml, GRID_COMBINATIONS, TEST_SPLITS, RESAMPLING_RATIOS)
    grid_results.to_csv(CSV_GRID, index=False, encoding='utf-8-sig')

    # ----- PART 3: Bootstrap Robustness (only for combos E and F) -----
    print("[Step 3] Running bootstrap robustness (200 iterations)...")
    robust_results = run_robustness_analysis(df_ml, ROBUST_COMBOS, RESAMPLING_RATIOS)
    robust_results.to_csv(CSV_ROBUST, index=False, encoding='utf-8-sig')

    # ----- Write comprehensive report -----
    with open(TXT_REPORT, 'w', encoding='utf-8') as f:
        write_academic_report(filter_results, grid_results, robust_results, f)

    print(f"\n✅ Pipeline completed successfully.")
    print(f"   Full report: {TXT_REPORT}")
    print(f"   CSV outputs: {CSV_FILTER}, {CSV_GRID}, {CSV_ROBUST}")


if __name__ == "__main__":
    main()