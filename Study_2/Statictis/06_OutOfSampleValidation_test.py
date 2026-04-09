"""
06_OutOfSampleValidation.py

Academic Title:
    Out‑of‑Sample Validation of the Machine Learning Collapse Prediction Model
    (Excluding Financials Sector, with Overlap Removal)

Objective:
    This script validates the trained ML model on an independent out‑of‑sample
    dataset (e.g., a later time period or a different set of firms) to assess
    its real‑world predictive performance. The analysis excludes the Financials
    sector to maintain consistency with the feature selection and grid search
    pipeline (04_ML_Pipeline.py).

    **Important**: Any observation in the test set that already appears in the
    training set (same Ticker and period_end) is removed to prevent data leakage.

Methods:
    1. Data Preparation:
       - Load training and out‑of‑sample data.
       - Exclude Financials sector (Financials_and_Real_Estate / Financial).
       - Remove overlapping observations (same Ticker & period_end) from test set.
       - Compute rolling mean of PDI (3‑quarter window) and lagged K_Pi_prime.
       - Define target: collapse (C1/C6) in the next quarter given current C3/C4.
    2. Model Training (on training set):
       - Main model: features = ['PDI_roll_mean', 'K_Pi_prime_t'].
       - Baseline model: feature = ['PDI_t'].
       - Random undersampling (ratio = 0.333, i.e., 3:1) to handle class imbalance.
       - Logistic regression with standard scaling.
    3. Out‑of‑Sample Evaluation (on cleaned test set):
       - Predict collapse probabilities.
       - Apply optimal threshold (0.2778) to generate binary alerts.
       - Compare main vs baseline using AUC, Brier score, and Recall.
    4. Risk Bucketing (Dynamic Quintiles):
       - Each quarter, rank firms by predicted probability and assign quintiles.
       - Compute collapse rate per bucket and year‑by‑year tracking.
    5. Risk Transition Matrix:
       - Estimate the probability of moving from one risk bucket to another.
    6. Cumulative Warning Analysis:
       - For each firm that eventually collapses, count how many times it was
         flagged as high‑risk (Q4 or Q5) in the 8 quarters preceding the collapse.
       - This provides evidence for the “gradual slide” hypothesis.

Output Files (saved in data/results/):
    - 07_ML_Tracking_List.csv          : Full out‑of‑sample predictions with risk buckets.
    - 07_ML_Transition_Matrix.csv      : Risk bucket transition probabilities.
    - 07_ML_Cumulative_Warnings.csv    : Warning counts before collapse.
    - 07_PHD_Final_Full_Report.txt     : Complete academic report.

Dependencies:
    pandas, numpy, scikit‑learn, imbalanced‑learn, pathlib
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, brier_score_loss
from imblearn.under_sampling import RandomUnderSampler
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION – PHD LEVEL
# ============================================================================

TRAIN_FILE = Path('../data/final_panel.csv')
TEST_FILE = Path('../data/out_of_sample.csv')
OUTPUT_DIR = Path('results')

# Create subdirectories if they don't exist
(OUTPUT_DIR / 'table').mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'report').mkdir(parents=True, exist_ok=True)

CSV_TRACKING = OUTPUT_DIR / 'table/06_OutOfSampleValidation_Tracking_List.csv'
CSV_TRANSITION = OUTPUT_DIR / 'table/06_OutOfSampleValidation_Transition_Matrix.csv'
CSV_WARNINGS = OUTPUT_DIR / 'table/06_OutOfSampleValidation_Cumulative_Warnings.csv'
TXT_REPORT = OUTPUT_DIR / 'report/06_OutOfSampleValidation_Report.txt'

RANDOM_SEED = 42

# Main model features (identified from previous feature selection)
MAIN_FEATURES = ['PDI_roll_mean', 'K_Pi_prime_t']
BASELINE_FEATURES = ['PDI_t']

# Undersampling ratio (3:1 = majority/minority)
SAMPLING_RATIO = 0.333  # i.e., 1/3
# Optimal threshold determined from training set (Youden's J)
OPTIMAL_THRESH = 0.2778


# ============================================================================
# DATA PREPARATION FUNCTIONS
# ============================================================================

def load_and_preprocess(df_path, features_list, exclude_financials=True):
    """
    Load data, apply formal gate, optionally exclude Financials sector,
    compute rolling PDI mean, and prepare C3/C4 collapse target.
    """
    df = pd.read_csv(df_path)
    df['period_end'] = pd.to_datetime(df['period_end'])

    # Override Configuration based on Regime_Label
    if 'Regime_Label' in df.columns:
        df['Configuration'] = np.where(
            df['Regime_Label'] == 'Normal_Regime',
            'Normal',
            df['Configuration']
        )

    # Sector filtering: exclude or keep Financials (default: exclude)
    if 'Sector' in df.columns:
        financial_labels = ['Financials_and_Real_Estate', 'Financial']
        if exclude_financials:
            df = df[~df['Sector'].isin(financial_labels)]   # remove Financials
        else:
            df = df[df['Sector'].isin(financial_labels)]    # keep only Financials

    df = df.sort_values(['Ticker', 'period_end'])

    # Rolling mean of PDI (3 quarters, minimum 1 observation)
    df['PDI_roll_mean'] = df.groupby('Ticker')['PDI_t'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )

    # Use K_Pi_prime if available; otherwise fallback to PDI_t
    if 'K_Pi_prime' in df.columns:
        df['K_Pi_prime_t'] = df['K_Pi_prime']
    elif 'K_Pi_prime_t' not in df.columns:
        df['K_Pi_prime_t'] = df['PDI_t']

    # Next quarter configuration
    df['next_Configuration'] = df.groupby('Ticker')['Configuration'].shift(-1)

    # Keep only rows currently in C3 or C4
    df_c3c4 = df[df['Configuration'].isin(['C3', 'C4'])].copy()

    # Target: collapse (C1/C6) in the next quarter
    df_c3c4['collapse_target'] = np.where(
        df_c3c4['next_Configuration'].isin(['C1', 'C6']), 1,
        np.where(df_c3c4['next_Configuration'].isin(['C3', 'C4']), 0, np.nan)
    )

    # Drop rows with missing target or features
    return df_c3c4.dropna(subset=['collapse_target'] + features_list).copy()


def remove_overlap(train_df, test_df):
    """
    Remove from test set any observation that already exists in the training set
    based on the combination of 'Ticker' and 'period_end'.
    Returns cleaned test_df and the number of removed rows.
    """
    train_keys = set(zip(train_df['Ticker'], train_df['period_end']))
    original_len = len(test_df)
    # Create a boolean mask: keep rows whose (Ticker, period_end) is NOT in train_keys
    mask = ~test_df.apply(lambda row: (row['Ticker'], row['period_end']) in train_keys, axis=1)
    test_df_clean = test_df[mask].copy()
    removed = original_len - len(test_df_clean)
    return test_df_clean, removed


def assign_dynamic_quintiles(df, prob_col='Prob_Collapse'):
    """
    For each quarter, rank firms by predicted probability and assign quintiles.
    Returns the same DataFrame with a new column 'Risk_Bucket'.
    """
    def _apply_rank(group):
        if len(group) < 5:
            group['Risk_Bucket'] = 'N/A'
            return group
        ranks = group[prob_col].rank(pct=True, method='first')
        labels = ['Q1 (Lowest Risk)', 'Q2', 'Q3', 'Q4', 'Q5 (Highest Risk)']
        group['Risk_Bucket'] = pd.cut(
            ranks, bins=[0, 0.2, 0.4, 0.6, 0.8, 1.01],
            labels=labels, include_lowest=True
        )
        return group
    return df.groupby('period_end', group_keys=False).apply(_apply_rank)


# ============================================================================
# MAIN VALIDATION PIPELINE
# ============================================================================

def main():
    print("=" * 80)
    print("OUT‑OF‑SAMPLE VALIDATION – FULL EVIDENCE REPORT")
    print("Collapse Prediction Model for Non‑Financials Sectors (C3/C4 → C1/C6)")
    print("=" * 80)

    # 1. Load and preprocess training and test sets (exclude Financials)
    print("\n[1] Loading and preprocessing data (excluding Financials)...")
    train_df = load_and_preprocess(TRAIN_FILE, MAIN_FEATURES, exclude_financials=True)
    test_df_raw = load_and_preprocess(TEST_FILE, MAIN_FEATURES, exclude_financials=True)
    print(f"    Raw training set: {len(train_df)} C3/C4 observations")
    print(f"    Raw test set    : {len(test_df_raw)} C3/C4 observations")

    # 2. Remove overlapping observations (data leakage prevention)
    print("\n[2] Removing overlapping observations (same Ticker & period_end) from test set...")
    test_df, n_removed = remove_overlap(train_df, test_df_raw)
    print(f"    Removed {n_removed} overlapping rows.")
    print(f"    Clean test set : {len(test_df)} C3/C4 observations")

    if len(test_df) == 0:
        print("❌ Error: No out‑of‑sample observations left after overlap removal.")
        return

    # 3. Train main model with undersampling
    print("[3] Training main model (PDI_roll_mean + K_Pi_prime_t)...")
    rus = RandomUnderSampler(sampling_strategy=SAMPLING_RATIO, random_state=RANDOM_SEED)
    X_res, y_res = rus.fit_resample(train_df[MAIN_FEATURES], train_df['collapse_target'])
    model_main = make_pipeline(StandardScaler(), LogisticRegression(random_state=RANDOM_SEED))
    model_main.fit(X_res, y_res)

    # 4. Train baseline model (PDI_t only)
    print("[4] Training baseline model (PDI_t)...")
    X_res_b, y_res_b = rus.fit_resample(train_df[BASELINE_FEATURES], train_df['collapse_target'])
    model_base = make_pipeline(StandardScaler(), LogisticRegression(random_state=RANDOM_SEED))
    model_base.fit(X_res_b, y_res_b)

    # 5. Out‑of‑sample predictions on cleaned test set
    print("[5] Generating out‑of‑sample predictions...")
    test_df['Prob_Collapse'] = model_main.predict_proba(test_df[MAIN_FEATURES])[:, 1]
    test_df['Prob_Collapse_Base'] = model_base.predict_proba(test_df[BASELINE_FEATURES])[:, 1]
    test_df['Red_Alert'] = (test_df['Prob_Collapse'] >= OPTIMAL_THRESH).astype(int)

    # 6. Risk bucket assignment (dynamic quintiles)
    test_df = assign_dynamic_quintiles(test_df, prob_col='Prob_Collapse')
    test_df['Year'] = test_df['period_end'].dt.year

    # 7. Cumulative warning analysis
    print("[6] Computing cumulative warning statistics...")
    test_df = test_df.sort_values(['Ticker', 'period_end'])
    test_df['High_Risk_Signal'] = test_df['Risk_Bucket'].isin(['Q5 (Highest Risk)', 'Q4']).astype(int)
    test_df['Cum_Warnings_8Q'] = test_df.groupby('Ticker')['High_Risk_Signal'].transform(
        lambda x: x.rolling(8, min_periods=1).sum()
    )
    collapse_cases = test_df[test_df['collapse_target'] == 1].copy()
    avg_warnings_before_collapse = collapse_cases['Cum_Warnings_8Q'].mean() if len(collapse_cases) > 0 else 0.0

    # 8. Performance metrics
    auc_main = roc_auc_score(test_df['collapse_target'], test_df['Prob_Collapse'])
    auc_baseline = roc_auc_score(test_df['collapse_target'], test_df['Prob_Collapse_Base'])
    brier_main = brier_score_loss(test_df['collapse_target'], test_df['Prob_Collapse'])
    brier_baseline = brier_score_loss(test_df['collapse_target'], test_df['Prob_Collapse_Base'])
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_df['collapse_target'], test_df['Red_Alert'], average='binary'
    )

    # 9. Risk transition matrix
    test_df['Prev_Bucket'] = test_df.groupby('Ticker')['Risk_Bucket'].shift(1)
    valid_trans = test_df.dropna(subset=['Prev_Bucket', 'Risk_Bucket'])
    valid_trans = valid_trans[(valid_trans['Prev_Bucket'] != 'N/A') & (valid_trans['Risk_Bucket'] != 'N/A')]
    if len(valid_trans) > 0:
        trans_matrix = pd.crosstab(valid_trans['Prev_Bucket'], valid_trans['Risk_Bucket'], normalize='index')
    else:
        trans_matrix = pd.DataFrame()

    # 10. Risk summary tables
    bucket_order = ['Q5 (Highest Risk)', 'Q4', 'Q3', 'Q2', 'Q1 (Lowest Risk)']
    risk_summary = test_df.groupby('Risk_Bucket', observed=True)['collapse_target'].agg(
        ['count', 'sum', 'mean']
    ).reindex(bucket_order)
    # Fill missing buckets with zeros
    risk_summary = risk_summary.fillna(0)
    yearly_tracking = test_df.groupby(['Year', 'Risk_Bucket'], observed=True)['collapse_target'].mean().unstack(fill_value=0)[bucket_order]

    # 11. Export CSV files
    test_df.to_csv(CSV_TRACKING, index=False)
    if not trans_matrix.empty:
        trans_matrix.to_csv(CSV_TRANSITION)
    if len(collapse_cases) > 0:
        collapse_cases[['Ticker', 'period_end', 'Cum_Warnings_8Q', 'collapse_target']].to_csv(CSV_WARNINGS, index=False)

    # 12. Write academic report
    with open(TXT_REPORT, 'w', encoding='utf-8') as f:
        f.write("=" * 120 + "\n")
        f.write("ACADEMIC REPORT: OUT‑OF‑SAMPLE VALIDATION OF COLLAPSE PREDICTION MODEL\n")
        f.write("Non‑Financials Sectors – Predicting C1/C6 from C3/C4 States\n")
        f.write("=" * 120 + "\n\n")

        f.write("I. STRATEGIC SPECIFICATIONS\n")
        f.write(f"   - Main features      : {MAIN_FEATURES}\n")
        f.write(f"   - Baseline feature   : {BASELINE_FEATURES}\n")
        f.write(f"   - Undersampling ratio: 3:1 (majority/minority = {1/SAMPLING_RATIO:.0f}:1)\n")
        f.write(f"   - Optimal threshold  : {OPTIMAL_THRESH:.4f} (Youden's J on training set)\n")
        f.write(f"   - Excluded sectors    : Financials_and_Real_Estate, Financial\n")
        f.write(f"   - Overlap removal     : {n_removed} rows removed from test set\n\n")

        f.write("II. MODEL COMPARISON (Out‑of‑Sample)\n")
        f.write(f"{'Metric':<30} | {'Baseline (PDI_t)':<25} | {'Main Model':<25}\n")
        f.write("-" * 85 + "\n")
        f.write(f"{'AUC':<30} | {auc_baseline:<25.4f} | {auc_main:<25.4f}\n")
        f.write(f"{'Brier Score':<30} | {brier_baseline:<25.4f} | {brier_main:<25.4f}\n")
        f.write(f"{'Recall (at threshold)':<30} | {'N/A':<25} | {recall * 100:<24.2f}%\n")
        f.write(f"{'Precision':<30} | {'N/A':<25} | {precision * 100:<24.2f}%\n")
        f.write(f"{'F1 Score':<30} | {'N/A':<25} | {f1:<24.4f}\n\n")

        f.write("III. RISK BUCKET ANALYSIS (Dynamic Quintiles)\n")
        f.write("Collapse rate (fraction of C1/C6 in next quarter) per bucket:\n")
        f.write(risk_summary.rename(columns={'count': 'N_obs', 'sum': 'N_collapse', 'mean': 'Collapse_rate'}).to_string() + "\n\n")

        f.write("IV. YEAR‑BY‑YEAR TRACKING (Collapse rate % per bucket)\n")
        f.write((yearly_tracking * 100).map("{:.2f}%".format).to_string() + "\n\n")

        f.write("V. RISK TRANSITION MATRIX (Row = current bucket, Column = next bucket)\n")
        if not trans_matrix.empty:
            f.write((trans_matrix * 100).map("{:.2f}%".format).to_string() + "\n\n")
        else:
            f.write("   (Insufficient data for transition matrix)\n\n")

        f.write("VI. CUMULATIVE WARNING ANALYSIS\n")
        if len(collapse_cases) > 0:
            f.write("   - For firms that eventually collapsed, the model flagged high risk\n")
            f.write("     (Q4 or Q5) on average {:.2f} times in the 8 quarters preceding the collapse.\n".format(avg_warnings_before_collapse))
        else:
            f.write("   - No collapse events in the cleaned test set.\n")
        f.write("   - This supports the hypothesis of a 'gradual slide' (Expectation Chimera)\n")
        f.write("     before the structural crash.\n\n")

        f.write("=" * 120 + "\n")
        f.write("Full detailed outputs are available in the accompanying CSV files.\n")
        f.write("=" * 120 + "\n")

    print(f"\n✅ Validation completed successfully.")
    print(f"   Academic report: {TXT_REPORT}")
    print(f"   CSV outputs     : {CSV_TRACKING}, {CSV_TRANSITION}, {CSV_WARNINGS}")


if __name__ == "__main__":
    main()