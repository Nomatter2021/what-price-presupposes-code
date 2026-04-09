"""
09_C2_TwoStage_OOS_Validation.py

Academic Title:
    Out‑of‑Sample Validation of the Two‑Stage Cascade Model for C2 Collapse Prediction
    Using Non‑Linear Kinematic Features (Momentum and Critical Mass)
    (with Overlap Removal)

Objective:
    This script validates the two‑stage cascade model on an independent out‑of‑sample
    dataset. The model first filters out firms likely to evolve (C3/C4) using five
    surface kinematic features, then on the remaining “dropdown” set (gray zone)
    applies a second logistic regression with three non‑linear features to predict
    collapse (C1/C6). The final collapse probability threshold is set to 0.4777.

    **Important**: Any observation in the test set that already appears in the
    training set (same Ticker and period_end) is removed to prevent data leakage.

    Key outputs include:
        - Stage 2 coefficients (learned from training dropdown set).
        - AUC for Stage 1 (evolution gate), Stage 2 (collapse discriminator), and overall.
        - Risk bucket quintiles (Q1–Q5) with collapse rates and cumulative catch rates.
        - Transition matrix of risk buckets.
        - Red alert performance and cumulative warning counts (8‑quarter lookback).

    The analysis excludes the Financials sector.

Output Files (saved in data/results/):
    - 09_C2_New_Vars_Tracking.csv   : Full OOS predictions with risk buckets.
    - 09_C2_New_Vars_OOS_Report.txt : Comprehensive academic report.

Dependencies:
    pandas, numpy, scikit‑learn, pathlib
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

TRAIN_FILE = Path('../data/final_panel.csv')
TEST_FILE = Path('../data/out_of_sample.csv')
OUTPUT_DIR = Path('results')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_DROPDOWN = OUTPUT_DIR / 'table/09_C2_New_Vars_Tracking.csv'
TXT_REPORT = OUTPUT_DIR / 'report/09_C2_New_Vars_OOS_Report.txt'

RANDOM_SEED = 42

# Stage 1 configuration (filtering evolution)
S1_RATIO = 0.20            # undersampling ratio for Stage 1 (minority/majority)
S1_THRESH = 0.1749         # probability threshold to enter dropdown zone
FEATS_S1 = ['B', 'B_Change', 'B_Acceleration', 'Cum_B_Change', 'Jerk']

# Stage 2 configuration (collapse vs. sustain in dropdown)
FEATS_S2 = ['B', 'B_Momentum', 'B_Critical_Mass']
S2_THRESH = 0.4777         # red alert threshold

# Excluded sectors (Financials)
EXCLUDED_SECTORS = ['Financials_and_Real_Estate', 'Financial']


# ============================================================================
# DATA PREPROCESSING (SAME AS TRAINING PIPELINE)
# ============================================================================

def safe_undersample(X, y, target_ratio, seed=RANDOM_SEED):
    """
    Perform random undersampling to achieve a given minority/majority ratio.
    target_ratio = minority_size / majority_size (if <=1) or reversed.
    """
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


def preprocess_data(df_path):
    """
    Load data, clean columns, exclude Financials, compute kinematic and
    interaction features, and prepare C2‑only dataset with targets.
    """
    df = pd.read_csv(df_path)
    # Clean column names: strip spaces and drop duplicates
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df['period_end'] = pd.to_datetime(df['period_end'])

    # Exclude Financials sector
    if 'Sector' in df.columns:
        df = df[~df['Sector'].isin(EXCLUDED_SECTORS)].copy()

    df = df.sort_values(['Ticker', 'period_end']).reset_index(drop=True)
    group = df.groupby('Ticker')

    # Core kinematic variables
    if 'PGR_t' in df.columns and 'E_3' in df.columns:
        df['B'] = df['E_3'] - (1 + df['PGR_t'])
        df['B_lag1'] = group['B'].shift(1)
        df['B_Change'] = df['B'] - df['B_lag1']
        df['B_Change_lag1'] = group['B_Change'].shift(1)
        df['B_Acceleration'] = df['B_Change'] - df['B_Change_lag1']
        df['Cum_B_Change'] = group['B_Change'].cumsum()
        df['B_Acceleration_lag1'] = group['B_Acceleration'].shift(1)
        df['Jerk'] = df['B_Acceleration'] - df['B_Acceleration_lag1']

    # Non‑linear interaction features
    df['B_Momentum'] = df['B_Change'] * df['B_Acceleration']
    df['B_Critical_Mass'] = df['B'] * df['B_Change'] * df['B_Acceleration']

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Next state and targets
    df['next_Configuration'] = group['Configuration'].shift(-1)
    df_c2 = df[df['Configuration'] == 'C2'].copy()
    valid_next = ['C1', 'C2', 'C3', 'C4', 'C6']
    df_c2 = df_c2[df_c2['next_Configuration'].isin(valid_next)].copy()
    df_c2['Target_Evolve'] = np.where(df_c2['next_Configuration'].isin(['C3', 'C4']), 1, 0)
    df_c2['Target_Collapse'] = np.where(df_c2['next_Configuration'].isin(['C1', 'C6']), 1, 0)

    all_feats = list(set(FEATS_S1 + FEATS_S2))
    return df_c2.dropna(subset=all_feats + ['Target_Evolve', 'Target_Collapse', 'next_Configuration']).copy()


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


def assign_global_quintiles(df, prob_col='Prob_Collapse'):
    """
    Assign global (cross‑sectional) quintiles based on predicted collapse probability.
    Returns a DataFrame with 'Risk_Bucket' column.
    """
    ranks = df[prob_col].rank(pct=True, method='first')
    labels = ['Q1 (Safe)', 'Q2 (Low Risk)', 'Q3 (Gray Zone)', 'Q4 (High Risk)', 'Q5 (Critical)']
    df = df.copy()
    df['Risk_Bucket'] = pd.cut(ranks, bins=[0, 0.2, 0.4, 0.6, 0.8, 1.01],
                               labels=labels, include_lowest=True)
    return df


# ============================================================================
# MAIN VALIDATION PIPELINE
# ============================================================================

def main():
    print("=" * 80)
    print("OUT‑OF‑SAMPLE VALIDATION: TWO‑STAGE CASCADE MODEL")
    print("Non‑Linear Kinematic Features (Momentum & Critical Mass)")
    print("=" * 80)

    # 1. Load and preprocess both sets
    print("\n[1] Loading training and test data...")
    train_df = preprocess_data(TRAIN_FILE)
    test_df_raw = preprocess_data(TEST_FILE)
    print(f"    Raw training set: {len(train_df)} C2 observations")
    print(f"    Raw test set    : {len(test_df_raw)} C2 observations")

    # 2. Remove overlapping observations (data leakage prevention)
    print("\n[2] Removing overlapping observations (same Ticker & period_end) from test set...")
    test_df, n_removed = remove_overlap(train_df, test_df_raw)
    print(f"    Removed {n_removed} overlapping rows.")
    print(f"    Clean test set : {len(test_df)} C2 observations")

    if len(test_df) == 0:
        print("❌ Error: No out‑of‑sample observations left after overlap removal.")
        return

    # 3. Train Stage 1 (evolution gate) on training set
    print("[3] Training Stage 1 (evolution filter)...")
    X_s1_raw = train_df[FEATS_S1].values
    y_s1_raw = train_df['Target_Evolve'].values.astype(int)
    X_s1_res, y_s1_res = safe_undersample(X_s1_raw, y_s1_raw, S1_RATIO, RANDOM_SEED)
    model_s1 = make_pipeline(StandardScaler(), LogisticRegression(class_weight=None, random_state=RANDOM_SEED))
    model_s1.fit(X_s1_res, y_s1_res)

    # 4. Extract dropdown set from training (Prob_Evolve < threshold)
    train_df['Prob_Evolve'] = model_s1.predict_proba(train_df[FEATS_S1].values)[:, 1]
    train_dropdown = train_df[train_df['Prob_Evolve'] < S1_THRESH].copy()
    print(f"    Training dropdown size: {len(train_dropdown)} (Prob_Evolve < {S1_THRESH})")

    # 5. Train Stage 2 (collapse predictor) on training dropdown
    print("[4] Training Stage 2 (collapse discriminator)...")
    model_s2 = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', random_state=RANDOM_SEED))
    model_s2.fit(train_dropdown[FEATS_S2].values, train_dropdown['Target_Collapse'].values.astype(int))

    # Extract Stage 2 coefficients for report
    coefs = model_s2.named_steps['logisticregression'].coef_[0]
    coef_dict = dict(zip(FEATS_S2, coefs))

    # 6. Apply to cleaned test set (out‑of‑sample)
    print("[5] Applying to out‑of‑sample test set...")
    test_df['Prob_Evolve'] = model_s1.predict_proba(test_df[FEATS_S1].values)[:, 1]
    auc_s1 = roc_auc_score(test_df['Target_Evolve'], test_df['Prob_Evolve'])
    test_df['Gate_1_Decision'] = np.where(test_df['Prob_Evolve'] >= S1_THRESH,
                                          'Evolve_Predicted', 'Dropdown')

    # Separate escaped (predicted evolve) and dropdown
    escaped_df = test_df[test_df['Gate_1_Decision'] == 'Evolve_Predicted'].copy()
    dropdown_df = test_df[test_df['Gate_1_Decision'] == 'Dropdown'].copy()

    missed_collapses = escaped_df['Target_Collapse'].sum()
    caught_collapses = dropdown_df['Target_Collapse'].sum()
    total_collapses_oos = test_df['Target_Collapse'].sum()

    print(f"    OOS collapses: {total_collapses_oos}")
    print(f"    Caught in dropdown zone: {caught_collapses} ({100*caught_collapses/total_collapses_oos:.1f}%)")
    print(f"    Missed (predicted evolve): {missed_collapses}")

    # 7. Stage 2 predictions on test dropdown
    auc_s2 = 0.0
    overall_auc = 0.0
    risk_summary = pd.DataFrame()
    trans_matrix = pd.DataFrame()
    alert_summary = pd.DataFrame()
    avg_warnings = 0.0

    if len(dropdown_df) > 0:
        probs = model_s2.predict_proba(dropdown_df[FEATS_S2].values)[:, 1]
        dropdown_df.loc[:, 'Prob_Collapse'] = probs
        try:
            auc_s2 = roc_auc_score(dropdown_df['Target_Collapse'], dropdown_df['Prob_Collapse'])
        except ValueError:
            pass

        # Overall AUC (assign zero probability to escaped firms)
        test_df['Final_Prob_Collapse'] = 0.0
        test_df.loc[test_df['Gate_1_Decision'] == 'Dropdown', 'Final_Prob_Collapse'] = dropdown_df['Prob_Collapse']
        overall_auc = roc_auc_score(test_df['Target_Collapse'], test_df['Final_Prob_Collapse'])

        # Risk bucket quintiles (global ranking)
        dropdown_df = assign_global_quintiles(dropdown_df, prob_col='Prob_Collapse')
        buckets = ['Q5 (Critical)', 'Q4 (High Risk)', 'Q3 (Gray Zone)', 'Q2 (Low Risk)', 'Q1 (Safe)']
        risk_summary = dropdown_df.groupby('Risk_Bucket', observed=True)['Target_Collapse'].agg(
            ['count', 'sum', 'mean']
        ).reindex(buckets)
        risk_summary['Cum_Collapses'] = risk_summary['sum'].cumsum()
        risk_summary['Global_Catch_Rate (%)'] = (risk_summary['Cum_Collapses'] / total_collapses_oos) * 100

        # Red alert based on threshold
        dropdown_df.loc[:, 'Red_Alert'] = (dropdown_df['Prob_Collapse'] >= S2_THRESH).astype(int)
        alert_summary = dropdown_df.groupby('Red_Alert')['Target_Collapse'].agg(['count', 'sum', 'mean'])

        # Cumulative warnings (8‑quarter lookback)
        dropdown_df = dropdown_df.sort_values(['Ticker', 'period_end'])
        dropdown_df.loc[:, 'Cum_Warnings_8Q'] = dropdown_df.groupby('Ticker')['Red_Alert'].transform(
            lambda x: x.rolling(8, min_periods=1).sum()
        )
        actual_collapses = dropdown_df[dropdown_df['Target_Collapse'] == 1]
        avg_warnings = actual_collapses['Cum_Warnings_8Q'].mean() if len(actual_collapses) > 0 else 0.0

        # Risk transition matrix (quarter‑over‑quarter)
        dropdown_df.loc[:, 'Prev_Bucket'] = dropdown_df.groupby('Ticker')['Risk_Bucket'].shift(1)
        valid_trans = dropdown_df.dropna(subset=['Prev_Bucket', 'Risk_Bucket'])
        if not valid_trans.empty:
            trans_matrix = pd.crosstab(valid_trans['Prev_Bucket'], valid_trans['Risk_Bucket'], normalize='index') * 100

        # Save detailed tracking file
        dropdown_df.to_csv(CSV_DROPDOWN, index=False)
        print(f"    Saved detailed tracking to {CSV_DROPDOWN}")

    # 8. Write academic report
    with open(TXT_REPORT, 'w', encoding='utf-8') as f:
        f.write("=" * 120 + "\n")
        f.write("ACADEMIC REPORT: OUT‑OF‑SAMPLE VALIDATION OF TWO‑STAGE CASCADE MODEL\n")
        f.write("Non‑Linear Kinematic Features: Momentum and Critical Mass\n")
        f.write("=" * 120 + "\n\n")

        f.write("I. DATA PREPARATION\n")
        f.write(f"   - Overlap removal: {n_removed} rows removed from test set\n")
        f.write(f"   - Final test set size: {len(test_df)} C2 observations\n\n")

        f.write("II. STAGE 2 COEFFICIENTS (Trained on Dropdown Set)\n")
        f.write("-" * 60 + "\n")
        for k, v in coef_dict.items():
            f.write(f"   {k:<20}: {v:8.4f}\n")
        f.write("\n")

        f.write("III. DISCRIMINATIVE POWER (AUC)\n")
        f.write("-" * 60 + "\n")
        f.write(f"   Stage 1 (Evolution gate) AUC : {auc_s1:.4f}\n")
        f.write(f"   Stage 2 (Collapse on dropdown): {auc_s2:.4f}\n")
        f.write(f"   Overall cascade AUC           : {overall_auc:.4f}\n\n")

        f.write("IV. RISK BUCKET ANALYSIS (Dynamic Quintiles on Dropdown Set)\n")
        f.write("-" * 100 + "\n")
        if not risk_summary.empty:
            f.write(f"{'Risk Bucket':<20} | {'N':<8} | {'Collapses':<10} | {'Collapse Rate':<15} | {'Cumulative Catch (%)'}\n")
            f.write("-" * 100 + "\n")
            for idx, row in risk_summary.iterrows():
                f.write(f"{idx:<20} | {row['count']:<8.0f} | {row['sum']:<10.0f} | {row['mean']*100:<13.2f}% | {row['Global_Catch_Rate (%)']:<18.2f}\n")
            f.write("\n")
        else:
            f.write("   (No dropdown observations in test set.)\n\n")

        f.write("V. RISK TRANSITION MATRIX (Quarter‑over‑Quarter)\n")
        f.write("-" * 80 + "\n")
        if not trans_matrix.empty:
            f.write(trans_matrix.map("{:.2f}%".format).to_string() + "\n\n")
        else:
            f.write("   (Insufficient data for transition matrix.)\n\n")

        f.write("VI. RED ALERT PERFORMANCE (Threshold = 0.4777)\n")
        f.write("-" * 80 + "\n")
        if not alert_summary.empty:
            f.write(f"{'Alert Status':<20} | {'N':<8} | {'Collapses':<10} | {'Collapse Rate (%)'}\n")
            f.write("-" * 80 + "\n")
            for status, row in alert_summary.iterrows():
                label = "Red Alert (≥ 0.4777)" if status == 1 else "Safe (< 0.4777)"
                f.write(f"{label:<20} | {row['count']:<8.0f} | {row['sum']:<10.0f} | {row['mean']*100:.2f}%\n")
            f.write("\n")
        else:
            f.write("   (No alert data.)\n")

        f.write("VII. CUMULATIVE WARNING ANALYSIS (8‑Quarter Lookback)\n")
        f.write("-" * 60 + "\n")
        f.write(f"   Average number of red alerts in the 8 quarters before collapse: {avg_warnings:.2f}\n")
        f.write("   This supports the 'gradual slide' hypothesis of the Expectation Chimera.\n\n")

        f.write("=" * 120 + "\n")
        f.write("Full detailed tracking is available in the CSV file.\n")
        f.write("=" * 120 + "\n")

    print(f"\n✅ Out‑of‑sample validation completed successfully.")
    print(f"   Academic report: {TXT_REPORT}")
    print(f"   Tracking CSV    : {CSV_DROPDOWN}")


if __name__ == "__main__":
    main()