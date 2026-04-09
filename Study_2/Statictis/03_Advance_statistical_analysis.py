"""
03_AdvancedEconometrics.py

Academic Title:
    Advanced Econometric Analysis of Market Regime Dynamics:
    Panel Fixed Effects, Granger Causality, Outlier Detection,
    Temporal Stability, and Bootstrap Robustness

Objectives:
    This script provides rigorous econometric evidence for the relationships
    between key variables and crash events, using panel data methods and
    causality tests.

Methods Included:
    1. Panel Fixed Effects (Within Transformation):
       - E_3 → R_t
       - E_3 → PDI_t
       Estimates entity‑specific intercepts (firm or cycle) to control for
       unobserved heterogeneity.

    2. Granger Causality Tests (F‑test, lag = 1):
       - Does PDI_lag1 Granger‑cause Crash (is_Crash)?
       - Does Crash_lag1 Granger‑cause PDI_t?

    3. Cook's Distance Outlier Detection:
       - Identify influential observations in the regression of Crash on PDI_lag1.
       - Threshold: 4 / n.

    4. Temporal Split Stability Test:
       - Pre‑2020 vs. Post‑2020 subsamples.
       - Re‑estimate PDI leading indicator (C3/C4) and fixed effects (E_3 → R_t).

    5. Bootstrap Robustness (1,000 iterations):
       - Resample with replacement from C3/C4 quarters.
       - Compute proportion of Mann‑Whitney U tests with p < 0.05 for
         PDI leading indicator.

Output Files (saved in data/results/):
    - advanced_econometrics_report.txt  : Full academic report.
    - fixed_effects_results.csv         : FE coefficients, p‑values, R².
    - granger_results.csv               : Granger F‑statistics and p‑values.
    - cooks_outliers.csv                : Influential observations (Cook's D > 4/n).
    - temporal_split_results.csv        : Pre/Post 2020 comparison.
    - bootstrap_robustness.csv          : Bootstrap success rate.

Dependencies:
    pandas, numpy, scipy, scikit‑learn, pathlib
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_FILE = Path('../data/final_panel.csv')
OUTPUT_DIR = Path('results')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TXT_REPORT = OUTPUT_DIR / 'report/03_advanced_econometrics_report.txt'
CSV_FE = OUTPUT_DIR / 'table/03_fixed_effects_results.csv'
CSV_GRANGER = OUTPUT_DIR / 'table/03_granger_results.csv'
CSV_COOKS = OUTPUT_DIR / 'table/03_cooks_outliers.csv'
CSV_TEMP = OUTPUT_DIR / 'table/03_temporal_split_results.csv'
CSV_BOOT = OUTPUT_DIR / 'table/03_bootstrap_robustness.csv'

VALID_STATES = ['Normal', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
CRASH_STATES = ['C1', 'C6']
RANDOM_SEED = 42
N_BOOT = 1000
SPLIT_YEAR = 2020
COOKS_THRESHOLD_FACTOR = 4  # threshold = factor / n
MIN_OBS_FOR_TEST = 3

np.random.seed(RANDOM_SEED)


# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================

def load_and_prepare_data():
    """
    Load the panel dataset, apply the formal gate, compute derived variables,
    and create lagged/lead variables for econometric analysis.
    """
    df = pd.read_csv(DATA_FILE)
    df['period_end'] = pd.to_datetime(df['period_end'])
    
    if 'Regime_Label' in df.columns:
        df['Configuration'] = np.where(
            df['Regime_Label'] == 'Normal_Regime',
            'Normal',
            df['Configuration']
        )
    
    # Keep only valid configurations
    df = df[df['Configuration'].isin(VALID_STATES)].copy()
    
    # Ensure numeric types
    numeric_cols = ['E_3', 'R_t', 'PDI_t', 'PGR_t', 'K_Pi_prime']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.sort_values(['Ticker', 'period_end']).reset_index(drop=True)
    
    # Define grouping column (Cycle_ID preferred)
    group_col = 'Cycle_ID' if 'Cycle_ID' in df.columns else 'Ticker'
    
    # Create crash dummy and lead/lag variables
    df['is_Crash'] = df['Configuration'].isin(CRASH_STATES).astype(int)
    df['Crash_next'] = df.groupby(group_col)['is_Crash'].shift(-1)
    df['PDI_lag1'] = df.groupby(group_col)['PDI_t'].shift(1)
    
    print(f"Data loaded: {len(df):,} observations.")
    return df, group_col


# ============================================================================
# 1. PANEL FIXED EFFECTS (WITHIN TRANSFORMATION)
# ============================================================================

def fixed_effects_regression(df, y_col, x_col, entity_col='Ticker'):
    """
    Estimate a panel fixed effects model using the within transformation.
    Returns a dictionary with coefficient, p‑value, number of observations, and R².
    """
    temp = df.dropna(subset=[y_col, x_col, entity_col]).copy()
    if len(temp) < 10:
        return None
    
    # Demean within each entity
    temp['y_demean'] = temp[y_col] - temp.groupby(entity_col)[y_col].transform('mean')
    temp['x_demean'] = temp[x_col] - temp.groupby(entity_col)[x_col].transform('mean')
    
    X = temp['x_demean'].values.reshape(-1, 1)
    y = temp['y_demean'].values
    
    if np.var(X) == 0:
        return None
    
    model = LinearRegression().fit(X, y)
    beta = model.coef_[0]
    
    # Compute p‑value (two‑tailed t‑test)
    residuals = y - model.predict(X)
    n = len(temp)
    k = 1  # number of predictors
    df_resid = n - k - 1
    se = np.sqrt(np.sum(residuals**2) / df_resid / np.sum((X - X.mean())**2))
    t_stat = beta / se
    p_val = 2 * (1 - stats.t.cdf(np.abs(t_stat), df_resid))
    
    return {
        'beta': beta,
        'p_val': p_val,
        'n_obs': n,
        'r2': model.score(X, y)
    }


# ============================================================================
# 2. GRANGER CAUSALITY TEST (LAG = 1)
# ============================================================================

def granger_causality_test(df, y_col, x_lag_col, y_lag_col, max_lag=1):
    """
    Perform a Granger causality test (F‑test) with one lag.
    Tests whether x_lag_col Granger‑causes y_col, conditional on the lag of y.
    Returns F‑statistic, p‑value, and number of observations.
    """
    temp = df.dropna(subset=[y_col, x_lag_col, y_lag_col]).copy()
    if len(temp) < 10:
        return None
    
    # Unrestricted model: y_t = a + b1*y_{t-1} + c1*x_{t-1}
    X_unrestricted = np.column_stack((
        np.ones(len(temp)),
        temp[y_lag_col].values,
        temp[x_lag_col].values
    ))
    # Restricted model: y_t = a + b1*y_{t-1}
    X_restricted = np.column_stack((
        np.ones(len(temp)),
        temp[y_lag_col].values
    ))
    y = temp[y_col].values
    
    def sum_squared_residuals(X, y):
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ beta
        return np.sum(resid**2)
    
    ssr_r = sum_squared_residuals(X_restricted, y)
    ssr_ur = sum_squared_residuals(X_unrestricted, y)
    
    n = len(temp)
    k_ur = X_unrestricted.shape[1]  # number of unrestricted parameters
    q = 1  # number of restrictions (coefficient on x_lag)
    f_stat = ((ssr_r - ssr_ur) / q) / (ssr_ur / (n - k_ur))
    p_val = 1 - stats.f.cdf(f_stat, q, n - k_ur)
    
    return {
        'f_stat': f_stat,
        'p_val': p_val,
        'n_obs': n
    }


# ============================================================================
# 3. COOK'S DISTANCE FOR OUTLIER DETECTION
# ============================================================================

def cooks_distance_outliers(df, y_col, x_col, entity_col='Ticker'):
    """
    Compute Cook's distance for each observation in a simple linear regression
    of y_col on x_col. Identify influential observations with Cook's D > 4/n.
    Returns a DataFrame with the most influential rows.
    """
    temp = df.dropna(subset=[y_col, x_col, entity_col]).copy()
    if len(temp) < 5:
        return None
    
    X = np.column_stack((np.ones(len(temp)), temp[x_col].values))
    y = temp[y_col].values
    
    # OLS estimation
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    pred = X @ beta
    resid = y - pred
    k = X.shape[1]  # number of parameters (including intercept)
    n = len(y)
    mse = np.sum(resid**2) / (n - k)
    
    # Hat matrix diagonal (leverage)
    XTX_inv = np.linalg.inv(X.T @ X)
    h = np.diag(X @ XTX_inv @ X.T)
    
    # Cook's distance
    cooks = (resid**2 / (k * mse)) * (h / (1 - h)**2)
    temp['Cooks_D'] = cooks
    
    threshold = COOKS_THRESHOLD_FACTOR / n
    outliers = temp[temp['Cooks_D'] > threshold].sort_values('Cooks_D', ascending=False)
    
    if len(outliers) == 0:
        return None
    
    return outliers[[entity_col, 'period_end', 'Configuration', 'Cooks_D', x_col, y_col]]


# ============================================================================
# 4. TEMPORAL SPLIT STABILITY (PRE‑2020 VS POST‑2020)
# ============================================================================

def temporal_split_analysis(df, group_col, split_year=SPLIT_YEAR):
    """
    Split the sample at split_year and re‑estimate key tests:
        - PDI leading indicator in C3/C4 (Mann‑Whitney U)
        - Fixed effects of E_3 on R_t
    Returns a dictionary of results for each period.
    """
    df_pre = df[df['period_end'].dt.year < split_year].copy()
    df_post = df[df['period_end'].dt.year >= split_year].copy()
    
    results = {}
    for period_name, df_period in [('Pre-2020', df_pre), ('Post-2020', df_post)]:
        if len(df_period) < 50:
            results[period_name] = None
            continue
        
        # PDI leading indicator (same as in 02_DynamicsLeading)
        df_period['Crash_next'] = df_period.groupby(group_col)['is_Crash'].shift(-1)
        c3c4 = df_period[df_period['Configuration'].isin(['C3', 'C4'])].copy()
        crash_pdi = c3c4[c3c4['Crash_next'] == 1]['PDI_t'].dropna()
        safe_pdi = c3c4[c3c4['Crash_next'] == 0]['PDI_t'].dropna()
        if len(crash_pdi) >= MIN_OBS_FOR_TEST and len(safe_pdi) >= MIN_OBS_FOR_TEST:
            _, p_pdi = stats.mannwhitneyu(crash_pdi, safe_pdi, alternative='two-sided')
        else:
            p_pdi = np.nan
        
        # Fixed effects: E_3 → R_t
        fe_res = fixed_effects_regression(df_period, 'R_t', 'E_3')
        
        results[period_name] = {
            'p_pdi_leading': p_pdi,
            'fe_beta': fe_res['beta'] if fe_res else np.nan,
            'fe_p': fe_res['p_val'] if fe_res else np.nan,
            'n_obs': len(df_period)
        }
    
    return results


# ============================================================================
# 5. BOOTSTRAP ROBUSTNESS FOR PDI LEADING INDICATOR
# ============================================================================

def bootstrap_pdi_leading_robustness(df, group_col, n_iter=N_BOOT):
    """
    Bootstrap the Mann‑Whitney test for PDI leading in C3/C4.
    Returns the proportion of bootstrap samples where p < 0.05.
    """
    # Prepare base data
    df_base = df.copy()
    df_base['Crash_next'] = df_base.groupby(group_col)['is_Crash'].shift(-1)
    c3c4 = df_base[df_base['Configuration'].isin(['C3', 'C4'])].dropna(subset=['PDI_t', 'Crash_next'])
    
    if len(c3c4) == 0:
        return None
    
    success_count = 0
    for _ in range(n_iter):
        sample = c3c4.sample(frac=1.0, replace=True)  # bootstrap sample
        crash_pdi = sample[sample['Crash_next'] == 1]['PDI_t']
        safe_pdi = sample[sample['Crash_next'] == 0]['PDI_t']
        if len(crash_pdi) >= MIN_OBS_FOR_TEST and len(safe_pdi) >= MIN_OBS_FOR_TEST:
            _, p = stats.mannwhitneyu(crash_pdi, safe_pdi, alternative='two-sided')
            if p < 0.05:
                success_count += 1
    
    return success_count / n_iter


# ============================================================================
# ACADEMIC REPORT GENERATION
# ============================================================================

def write_academic_report(fe_results, granger_results, cooks_outliers,
                          temporal_results, bootstrap_rate, f):
    """
    Write the full academic report to the output file handle.
    """
    f.write("=" * 100 + "\n")
    f.write("ADVANCED ECONOMETRIC ANALYSIS – ACADEMIC REPORT\n")
    f.write("Panel Fixed Effects, Granger Causality, Outlier Detection,\n")
    f.write("Temporal Stability, and Bootstrap Robustness\n")
    f.write("=" * 100 + "\n\n")
    
    # ----- 1. Panel Fixed Effects -----
    f.write("1. PANEL FIXED EFFECTS (Within Transformation)\n")
    f.write("-" * 50 + "\n")
    if fe_results:
        for model_name, res in fe_results.items():
            if res:
                f.write(f"   Model: {model_name}\n")
                f.write(f"      Coefficient (β) = {res['beta']:.4f}\n")
                f.write(f"      p‑value          = {res['p_val']:.2e}\n")
                f.write(f"      Observations     = {res['n_obs']}\n")
                f.write(f"      R² (within)      = {res['r2']:.3f}\n")
                if res['p_val'] < 0.05:
                    f.write("      → Statistically significant at the 5% level.\n")
                else:
                    f.write("      → Not statistically significant.\n")
            else:
                f.write(f"   Model: {model_name} – insufficient data.\n")
    else:
        f.write("   No fixed effects models could be estimated.\n")
    
    # ----- 2. Granger Causality -----
    f.write("\n2. GRANGER CAUSALITY TESTS (F‑test, lag = 1)\n")
    f.write("-" * 50 + "\n")
    if granger_results:
        for test_name, res in granger_results.items():
            if res:
                f.write(f"   Hypothesis: {test_name}\n")
                f.write(f"      F‑statistic = {res['f_stat']:.4f}\n")
                f.write(f"      p‑value     = {res['p_val']:.2e}\n")
                f.write(f"      Observations = {res['n_obs']}\n")
                if res['p_val'] < 0.05:
                    f.write("      → Reject H₀: Granger causality exists.\n")
                else:
                    f.write("      → Fail to reject H₀: No Granger causality detected.\n")
            else:
                f.write(f"   Hypothesis: {test_name} – insufficient data.\n")
    else:
        f.write("   No Granger tests could be performed.\n")
    
    # ----- 3. Cook's Distance Outliers -----
    f.write("\n3. INFLUENTIAL OBSERVATIONS (Cook's Distance)\n")
    f.write("-" * 50 + "\n")
    f.write("   Regression: Crash (C1/C6) ~ PDI_lag1\n")
    if cooks_outliers is not None and len(cooks_outliers) > 0:
        f.write(f"   Number of influential observations (Cook's D > 4/n): {len(cooks_outliers)}\n")
        f.write("   Top 5 influential points:\n")
        for _, row in cooks_outliers.head(5).iterrows():
            ticker = row['Ticker']
            date = pd.to_datetime(row['period_end']).date()
            cooks_d = row['Cooks_D']
            f.write(f"      {ticker} – {date}: Cook's D = {cooks_d:.4f}\n")
    else:
        f.write("   No influential observations detected.\n")
    
    # ----- 4. Temporal Split Stability -----
    f.write("\n4. TEMPORAL STABILITY (Pre‑2020 vs. Post‑2020)\n")
    f.write("-" * 50 + "\n")
    if temporal_results:
        for period, res in temporal_results.items():
            if res:
                f.write(f"   {period} (n = {res['n_obs']}):\n")
                f.write(f"      PDI leading (C3/C4) p‑value = {res['p_pdi_leading']:.2e}\n")
                if not np.isnan(res['fe_beta']):
                    f.write(f"      Fixed effects (E_3 → R_t): β = {res['fe_beta']:.4f} (p = {res['fe_p']:.2e})\n")
                else:
                    f.write("      Fixed effects (E_3 → R_t): insufficient data.\n")
            else:
                f.write(f"   {period}: insufficient data.\n")
    else:
        f.write("   Temporal split not possible.\n")
    
    # ----- 5. Bootstrap Robustness -----
    f.write("\n5. BOOTSTRAP ROBUSTNESS (PDI Leading Indicator)\n")
    f.write("-" * 50 + "\n")
    f.write(f"   Number of bootstrap iterations: {N_BOOT}\n")
    if bootstrap_rate is not None:
        f.write(f"   Proportion of iterations with p < 0.05: {bootstrap_rate * 100:.1f}%\n")
        if bootstrap_rate > 0.95:
            f.write("   → Interpretation: The PDI leading effect is highly robust.\n")
        elif bootstrap_rate > 0.80:
            f.write("   → Interpretation: The result is moderately robust.\n")
        else:
            f.write("   → Interpretation: The result is sensitive to sample variation.\n")
    else:
        f.write("   Bootstrap could not be performed (insufficient data).\n")
    
    f.write("\n" + "=" * 100 + "\n")
    f.write("Detailed numerical results are available in the accompanying CSV files.\n")
    f.write("=" * 100 + "\n")


# ============================================================================
# CSV EXPORT FUNCTIONS
# ============================================================================

def export_csv_files(fe_results, granger_results, cooks_outliers,
                     temporal_results, bootstrap_rate):
    """
    Export all results to CSV files.
    """
    # 1. Fixed effects
    fe_rows = []
    for model, res in fe_results.items():
        if res:
            fe_rows.append({
                'Model': model,
                'Beta': res['beta'],
                'P_value': res['p_val'],
                'N_obs': res['n_obs'],
                'R2': res['r2']
            })
    if fe_rows:
        pd.DataFrame(fe_rows).to_csv(CSV_FE, index=False, encoding='utf-8-sig')
    
    # 2. Granger causality
    granger_rows = []
    for test, res in granger_results.items():
        if res:
            granger_rows.append({
                'Test': test,
                'F_statistic': res['f_stat'],
                'P_value': res['p_val'],
                'N_obs': res['n_obs']
            })
    if granger_rows:
        pd.DataFrame(granger_rows).to_csv(CSV_GRANGER, index=False, encoding='utf-8-sig')
    
    # 3. Cook's outliers
    if cooks_outliers is not None and len(cooks_outliers) > 0:
        cooks_outliers.to_csv(CSV_COOKS, index=False, encoding='utf-8-sig')
    
    # 4. Temporal split
    temp_rows = []
    for period, res in temporal_results.items():
        if res:
            temp_rows.append({
                'Period': period,
                'N_obs': res['n_obs'],
                'PDI_leading_p': res['p_pdi_leading'],
                'FE_beta': res['fe_beta'],
                'FE_p': res['fe_p']
            })
    if temp_rows:
        pd.DataFrame(temp_rows).to_csv(CSV_TEMP, index=False, encoding='utf-8-sig')
    
    # 5. Bootstrap robustness
    if bootstrap_rate is not None:
        pd.DataFrame({'Bootstrap_success_rate': [bootstrap_rate]}).to_csv(
            CSV_BOOT, index=False, encoding='utf-8-sig'
        )
    
    print(f"All CSV files exported to {OUTPUT_DIR}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 80)
    print("ADVANCED ECONOMETRIC ANALYSIS")
    print("Fixed Effects, Granger Causality, Outlier Detection, and Robustness")
    print("=" * 80)
    
    # Load data
    df, group_col = load_and_prepare_data()
    
    # 1. Fixed Effects
    fe_results = {
        'R_t = α_i + β * E_3': fixed_effects_regression(df, 'R_t', 'E_3'),
        'PDI_t = α_i + β * E_3': fixed_effects_regression(df, 'PDI_t', 'E_3')
    }
    
    # 2. Granger Causality
    # Prepare data with complete lags
    df_granger = df.dropna(subset=['is_Crash', 'PDI_lag1', 'Crash_next'])
    granger_results = {
        'PDI_lag1 → Crash (is_Crash)': granger_causality_test(
            df_granger, 'is_Crash', 'PDI_lag1', 'Crash_next'
        ),
        'Crash_lag1 → PDI_t': granger_causality_test(
            df_granger, 'PDI_t', 'Crash_next', 'PDI_lag1'
        )
    }
    
    # 3. Cook's Distance
    cooks_outliers = cooks_distance_outliers(df, 'is_Crash', 'PDI_lag1')
    
    # 4. Temporal Split
    temporal_results = temporal_split_analysis(df, group_col, split_year=SPLIT_YEAR)
    
    # 5. Bootstrap Robustness
    bootstrap_rate = bootstrap_pdi_leading_robustness(df, group_col, n_iter=N_BOOT)
    
    # Write academic report
    with open(TXT_REPORT, 'w', encoding='utf-8') as f:
        write_academic_report(fe_results, granger_results, cooks_outliers,
                              temporal_results, bootstrap_rate, f)
    
    # Export CSV files
    export_csv_files(fe_results, granger_results, cooks_outliers,
                     temporal_results, bootstrap_rate)
    
    print(f"\n✅ Analysis completed successfully.")
    print(f"   Academic report: {TXT_REPORT}")
    print(f"   All CSV outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()