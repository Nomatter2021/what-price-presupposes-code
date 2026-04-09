"""
01_BasicDescriptives.py

Academic Title:
    Comprehensive Univariate and Multivariate Descriptive Analysis of
    Market Regimes (Normal, C1–C6)

Objectives:
    This script provides a complete descriptive and inferential statistical
    summary of the seven market configurations. The analysis is performed
    globally and for each sector (with at least 30 observations).

Methods Included:
    1. Distribution of configurations (counts and percentages).
    2. Descriptive statistics (mean, median, quartiles, etc.) for key variables:
       E_3, R_t, PDI_t, PGR_t, K_Pi_prime.
    3. Kruskal–Wallis H-test for overall differences among configurations.
    4. Pairwise Mann–Whitney U tests with Bonferroni correction.
    5. First‑order Markov transition matrix with bootstrap 95% CIs (1,000 resamples).
    6. Spearman rank correlations (E_3 vs R_t, E_3 vs PDI_t).
    7. Multivariate analysis:
        - MANOVA (Wilks' lambda) for (E_3, R_t, PDI_t).
        - Decision tree classification (max_depth=5) predicting configuration
          from E_3, R_t, PDI_t, with train/test split (70/30).

Output Files (saved in data/results/):
    - basic_descriptives_report.txt          : Full academic report.
    - basic_descriptives_config_distribution.csv
    - basic_descriptives_stats.csv
    - basic_descriptives_pairwise.csv
    - basic_descriptives_markov.csv
    - basic_descriptives_multivariate.csv

Dependencies:
    pandas, numpy, scipy, scikit‑learn, pathlib
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from itertools import combinations
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_FILE = Path('../data/final_panel.csv')
OUTPUT_DIR = Path('results')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TXT_REPORT = OUTPUT_DIR / 'report/01_basic_descriptives_report.txt'
CSV_CONFIG = OUTPUT_DIR / 'table/01_basic_descriptives_config_distribution.csv'
CSV_STATS = OUTPUT_DIR / 'table/01_basic_descriptives_stats.csv'
CSV_PAIRWISE = OUTPUT_DIR / 'table/01_basic_descriptives_pairwise.csv'
CSV_MARKOV = OUTPUT_DIR / 'table/01_basic_descriptives_markov.csv'
CSV_MULTIVAR = OUTPUT_DIR / 'table/01_basic_descriptives_multivariate.csv'

VALID_STATES = ['Normal', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
NUMERIC_VARS = ['E_3', 'R_t', 'PDI_t', 'PGR_t', 'K_Pi_prime']
N_BOOT = 1000
RANDOM_SEED = 42
TEST_SIZE = 0.3
MAX_DEPTH = 5

np.random.seed(RANDOM_SEED)


# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================

def load_and_prepare_data():
    """
    Load the panel dataset, apply the formal gate (Regime_Label overrides
    Configuration to 'Normal'), and filter to valid states.
    """
    df = pd.read_csv(DATA_FILE)
    df['period_end'] = pd.to_datetime(df['period_end'])
    
    if 'Regime_Label' in df.columns:
        df['Configuration'] = np.where(
            df['Regime_Label'] == 'Normal_Regime',
            'Normal',
            df['Configuration']
        )
    
    df = df[df['Configuration'].isin(VALID_STATES)].copy()
    
    for col in NUMERIC_VARS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.sort_values(['Ticker', 'period_end']).reset_index(drop=True)
    print(f"Data loaded: {len(df):,} observations.")
    return df


# ============================================================================
# UNIVARIATE INFERENCE
# ============================================================================

def kruskal_wallis_test(df, variable, group_col='Configuration'):
    """
    Perform Kruskal–Wallis H-test for differences in `variable` across all
    configuration groups present in the data.
    Returns (H_statistic, p_value, interpretation).
    """
    groups = [cfg for cfg in VALID_STATES if cfg in df[group_col].unique()]
    samples = [df[df[group_col] == cfg][variable].dropna() for cfg in groups]
    if len(samples) < 2:
        return np.nan, np.nan, 'Insufficient groups'
    h, p = stats.kruskal(*samples)
    interpretation = 'Significant' if p < 0.05 else 'Not significant'
    return h, p, interpretation


def pairwise_mannwhitney(df, variable, group_col='Configuration'):
    """
    Perform pairwise Mann–Whitney U tests between all configuration pairs.
    Applies Bonferroni correction for multiple comparisons.
    Returns a DataFrame with columns: Group1, Group2, U_statistic, P_value,
    P_adj, Significant_05, Significant_01.
    """
    present = [cfg for cfg in VALID_STATES if cfg in df[group_col].unique()]
    if len(present) < 2:
        return pd.DataFrame()
    
    results = []
    for g1, g2 in combinations(present, 2):
        s1 = df[df[group_col] == g1][variable].dropna()
        s2 = df[df[group_col] == g2][variable].dropna()
        if len(s1) >= 3 and len(s2) >= 3:
            try:
                u, p = stats.mannwhitneyu(s1, s2, alternative='two-sided')
                results.append({'Group1': g1, 'Group2': g2, 'U_statistic': u, 'P_value': p})
            except Exception:
                pass
    
    if not results:
        return pd.DataFrame()
    
    df_res = pd.DataFrame(results)
    n_comp = len(df_res)
    df_res['P_adj'] = (df_res['P_value'] * n_comp).clip(upper=1.0)
    df_res['Significant_05'] = df_res['P_adj'] < 0.05
    df_res['Significant_01'] = df_res['P_adj'] < 0.01
    return df_res


# ============================================================================
# MARKOV TRANSITION MATRIX WITH BOOTSTRAP CI
# ============================================================================

def markov_transition_matrix(df, group_col, states):
    """
    Compute the first-order Markov transition probability matrix (in percent)
    with bootstrap 95% confidence intervals (percentile method, N=1000).
    Returns a DataFrame with columns: From, To, Prob_pct, CI_lower_pct,
    CI_upper_pct, N_outgoing.
    """
    # Extract sequences per group (firm or cycle)
    sequences = df.groupby(group_col)['Configuration'].agg(list).tolist()
    sequences = [seq for seq in sequences if len(seq) >= 2]
    
    def compute_probs(seqs):
        counts = {s: {t: 0 for t in states} for s in states}
        totals = {s: 0 for s in states}
        for seq in seqs:
            for i in range(len(seq) - 1):
                s, t = seq[i], seq[i + 1]
                if s in counts and t in counts[s]:
                    counts[s][t] += 1
                    totals[s] += 1
        probs = {}
        for s in states:
            probs[s] = {}
            for t in states:
                probs[s][t] = counts[s][t] / totals[s] if totals[s] > 0 else 0.0
        return probs, counts, totals
    
    # Point estimates
    probs_orig, _, totals = compute_probs(sequences)
    
    # Bootstrap
    boot_probs = []
    n_seq = len(sequences)
    for _ in range(N_BOOT):
        boot_seqs = [sequences[i] for i in np.random.choice(n_seq, n_seq, replace=True)]
        p, _, _ = compute_probs(boot_seqs)
        boot_probs.append(p)
    
    def get_ci(probs_list, s, t):
        vals = [p[s][t] for p in probs_list]
        return np.percentile(vals, 2.5), np.percentile(vals, 97.5)
    
    rows = []
    for s in states:
        for t in states:
            prob_pct = probs_orig[s][t] * 100
            if totals.get(s, 0) >= 5:
                low, high = get_ci(boot_probs, s, t)
                ci_low, ci_high = low * 100, high * 100
            else:
                ci_low, ci_high = np.nan, np.nan
            rows.append({
                'From': s,
                'To': t,
                'Prob_pct': prob_pct,
                'CI_lower_pct': ci_low,
                'CI_upper_pct': ci_high,
                'N_outgoing': totals.get(s, 0)
            })
    return pd.DataFrame(rows)


# ============================================================================
# SPEARMAN CORRELATIONS
# ============================================================================

def spearman_correlations(df):
    """
    Compute Spearman's rank correlations:
        (1) E_3 vs R_t
        (2) E_3 vs PDI_t
    Returns (rho_E3_Rt, p_E3_Rt, rho_E3_PDI, p_E3_PDI).
    """
    rho_rt, p_rt = stats.spearmanr(df['E_3'], df['R_t'], nan_policy='omit')
    rho_pdi, p_pdi = stats.spearmanr(df['E_3'], df['PDI_t'], nan_policy='omit')
    return rho_rt, p_rt, rho_pdi, p_pdi


# ============================================================================
# MULTIVARIATE ANALYSIS: MANOVA AND DECISION TREE
# ============================================================================

def manova_wilks(df, dependent_vars, group_col='Configuration'):
    """
    Perform MANOVA using Wilks' lambda. Returns (Wilks_lambda, p_value).
    Implements the approximation by Rao (1951).
    """
    df_clean = df[dependent_vars + [group_col]].dropna()
    if len(df_clean) == 0:
        return np.nan, np.nan
    
    groups = df_clean[group_col].unique()
    if len(groups) < 2:
        return np.nan, np.nan
    
    n = len(df_clean)
    k = len(dependent_vars)
    overall_mean = df_clean[dependent_vars].mean().values.reshape(-1, 1)
    
    # Total sum of squares and cross-products (T)
    T = np.zeros((k, k))
    for _, row in df_clean.iterrows():
        x = row[dependent_vars].values.astype(float).reshape(-1, 1)
        dev = x - overall_mean
        T += dev @ dev.T
    
    # Within-group sum of squares (W)
    W = np.zeros((k, k))
    for g in groups:
        sub = df_clean[df_clean[group_col] == g][dependent_vars]
        if len(sub) < 2:
            continue
        mean_g = sub.mean().values.reshape(-1, 1)
        for _, row in sub.iterrows():
            x = row.values.astype(float).reshape(-1, 1)
            dev = x - mean_g
            W += dev @ dev.T
    
    try:
        wilks = np.linalg.det(W) / np.linalg.det(T)
    except np.linalg.LinAlgError:
        return np.nan, np.nan
    
    # Rao's approximation
    p = len(groups) - 1
    if p <= 0:
        return wilks, np.nan
    s = min(p, k)
    m = (abs(p - k) - 1) / 2
    N = n - 1
    r = N - (p - k + 1) / 2
    u = (r * s - 2) / (p * k)
    f_stat = (1 - wilks ** (1 / u)) / (wilks ** (1 / u)) * (r * s - 2) / (p * k)
    df1 = p * k
    df2 = r * s - 2
    if df2 > 0:
        p_val = 1 - stats.f.cdf(f_stat, df1, df2)
    else:
        p_val = np.nan
    return wilks, p_val


def decision_tree_classification(df, features, target, test_size=TEST_SIZE, max_depth=MAX_DEPTH):
    """
    Train a Decision Tree classifier to predict configuration from features.
    Returns:
        accuracy   : float
        report     : dict (classification_report output)
        cm         : confusion matrix (list of lists)
        importance : dict {feature: importance}
    """
    df_clean = df[features + [target]].dropna()
    if len(df_clean) == 0:
        return None, None, None, None
    
    X = df_clean[features]
    y = df_clean[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y
    )
    
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=RANDOM_SEED)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    accuracy = (y_pred == y_test).mean()
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    importance = dict(zip(features, clf.feature_importances_))
    
    return accuracy, report, cm, importance


# ============================================================================
# REPORT GENERATION (TEXT)
# ============================================================================

def write_academic_report(df_global, sector_dfs, pairwise_by_var, markov_global,
                          manova_res, class_res, f):
    """
    Write the full academic report to the output file handle.
    """
    f.write("=" * 100 + "\n")
    f.write("BASIC DESCRIPTIVES – ACADEMIC REPORT\n")
    f.write("Global Market and Sector-Level Analysis\n")
    f.write("=" * 100 + "\n\n")
    
    # ----- 1. Global -----
    f.write("1. GLOBAL MARKET\n")
    f.write("-" * 50 + "\n")
    
    # Distribution
    config_counts = df_global['Configuration'].value_counts()
    total = len(df_global)
    f.write("Configuration distribution:\n")
    for cfg in VALID_STATES:
        cnt = config_counts.get(cfg, 0)
        pct = cnt / total * 100
        f.write(f"   {cfg:6}: {cnt:6,} quarters ({pct:5.2f}%)\n")
    
    # Descriptive statistics (median + IQR)
    f.write("\nDescriptive statistics by configuration (median [Q1, Q3]):\n")
    for var in NUMERIC_VARS:
        if var not in df_global.columns:
            continue
        f.write(f"\n   {var}:\n")
        for cfg in VALID_STATES:
            vals = df_global[df_global['Configuration'] == cfg][var].dropna()
            if len(vals) > 0:
                med = vals.median()
                q1 = vals.quantile(0.25)
                q3 = vals.quantile(0.75)
                f.write(f"      {cfg:6}: n={len(vals):4}, median={med:8.4f} [Q1={q1:8.4f}, Q3={q3:8.4f}]\n")
            else:
                f.write(f"      {cfg:6}: n=0\n")
    
    # Kruskal–Wallis
    f.write("\nKruskal–Wallis H‑tests (differences across configurations):\n")
    for var in NUMERIC_VARS:
        if var not in df_global.columns:
            continue
        h, p, sig = kruskal_wallis_test(df_global, var)
        if not np.isnan(h):
            f.write(f"   {var:12}: H = {h:.2f}, p = {p:.2e} ({sig})\n")
        else:
            f.write(f"   {var:12}: insufficient data\n")
    
    # Pairwise (example for E_3 only, full in CSV)
    f.write("\nPairwise Mann–Whitney U tests (Bonferroni correction) – example for E_3:\n")
    if 'E_3' in pairwise_by_var:
        df_pair = pairwise_by_var['E_3']
        if not df_pair.empty:
            for _, row in df_pair.iterrows():
                sig_stars = "***" if row['Significant_01'] else ("**" if row['Significant_05'] else "")
                f.write(f"   {row['Group1']:6} vs {row['Group2']:6}: p‑adj = {row['P_adj']:.2e} {sig_stars}\n")
        else:
            f.write("   insufficient data\n")
    
    # Markov matrix
    f.write("\nMarkov transition matrix (percent, with 95% bootstrap CI if N_outgoing ≥5):\n")
    prob_mat = markov_global.pivot(index='From', columns='To', values='Prob_pct')
    ci_low_mat = markov_global.pivot(index='From', columns='To', values='CI_lower_pct')
    ci_high_mat = markov_global.pivot(index='From', columns='To', values='CI_upper_pct')
    f.write("\nFrom\\To   " + " ".join(f"{s:>7}" for s in VALID_STATES) + "\n")
    for s in VALID_STATES:
        row = f"{s:<8}"
        for t in VALID_STATES:
            prob = prob_mat.loc[s, t] if s in prob_mat.index and t in prob_mat.columns else 0.0
            if np.isnan(prob):
                prob = 0.0
            if (s in ci_low_mat.index and t in ci_low_mat.columns and
                not np.isnan(ci_low_mat.loc[s, t])):
                low = ci_low_mat.loc[s, t]
                high = ci_high_mat.loc[s, t]
                row += f" {prob:5.1f}[{low:3.0f}-{high:3.0f}]"
            else:
                row += f" {prob:5.1f}     "
        f.write(row + "\n")
    
    # Spearman correlations
    rho_rt, p_rt, rho_pdi, p_pdi = spearman_correlations(df_global)
    f.write("\nSpearman rank correlations (full sample):\n")
    f.write(f"   E_3 vs R_t   : ρ = {rho_rt:.4f}, p = {p_rt:.2e}\n")
    f.write(f"   E_3 vs PDI_t : ρ = {rho_pdi:.4f}, p = {p_pdi:.2e}\n")
    
    # ----- 2. Multivariate -----
    f.write("\n" + "=" * 100 + "\n")
    f.write("2. MULTIVARIATE ANALYSIS (E_3, R_t, PDI_t)\n")
    f.write("-" * 50 + "\n")
    
    if manova_res is not None:
        wilks, p_man = manova_res
        if not np.isnan(p_man):
            f.write(f"MANOVA (Wilks' lambda): Λ = {wilks:.4f}, p = {p_man:.2e}\n")
            if p_man < 0.001:
                f.write("   → Conclusion: The mean vector of (E_3, R_t, PDI_t) differs highly significantly across configurations.\n")
            else:
                f.write("   → Conclusion: No sufficient evidence of multivariate differences.\n")
        else:
            f.write("   MANOVA could not be computed (singular matrix or insufficient data).\n")
    else:
        f.write("   MANOVA not performed due to missing data.\n")
    
    if class_res is not None:
        acc, report, cm, importance = class_res
        f.write(f"\nDecision tree classification (70/30 train/test, max_depth={MAX_DEPTH}):\n")
        f.write(f"   Test set accuracy: {acc:.2%}\n")
        f.write("   Feature importance:\n")
        for feat, imp in importance.items():
            f.write(f"      - {feat}: {imp:.3f}\n")
        f.write("\n   Confusion matrix (rows = true, cols = predicted):\n")
        labels = [l for l in report.keys() if l not in ['accuracy', 'macro avg', 'weighted avg']]
        f.write("      " + " ".join(f"{l:>6}" for l in labels) + "\n")
        for i, true_lab in enumerate(labels):
            row_str = f"      {true_lab:6}"
            for j, _ in enumerate(labels):
                if i < len(cm) and j < len(cm[i]):
                    row_str += f" {cm[i][j]:6d}"
                else:
                    row_str += "      0"
            f.write(row_str + "\n")
        f.write("\n   F1‑scores by configuration:\n")
        for lab in labels:
            if lab in report:
                f.write(f"      {lab:6}: {report[lab]['f1-score']:.3f}\n")
    else:
        f.write("\n   Decision tree classification not possible (insufficient data).\n")
    
    # ----- 3. Sector summaries -----
    if sector_dfs:
        f.write("\n" + "=" * 100 + "\n")
        f.write("3. SECTOR‑LEVEL SUMMARIES (only sectors with ≥30 observations)\n")
        f.write("-" * 50 + "\n")
        for sector, df_sec in sector_dfs.items():
            f.write(f"\n--- {sector} (n = {len(df_sec):,}) ---\n")
            norm_cnt = (df_sec['Configuration'] == 'Normal').sum()
            f.write(f"   Normal proportion: {norm_cnt} ({norm_cnt/len(df_sec)*100:.1f}%)\n")
            h, p, sig = kruskal_wallis_test(df_sec, 'E_3')
            if not np.isnan(h):
                f.write(f"   Kruskal–Wallis (E_3): H = {h:.2f}, p = {p:.2e} ({sig})\n")
            wilks_s, p_s = manova_wilks(df_sec, ['E_3', 'R_t', 'PDI_t'])
            if not np.isnan(p_s):
                f.write(f"   MANOVA (E_3,R_t,PDI_t): p = {p_s:.2e}\n")
    
    f.write("\n" + "=" * 100 + "\n")
    f.write("Full detailed results (including all variables and all sectors) are available in the accompanying CSV files.\n")
    f.write("=" * 100 + "\n")


# ============================================================================
# CSV EXPORT FUNCTIONS
# ============================================================================

def export_all_csv(df_global, sector_dfs, all_pairwise, markov_global, markov_sectors,
                   class_global, class_sectors):
    """
    Export all DataFrames to CSV files in the results directory.
    """
    # 1. Configuration distribution
    rows_config = []
    total_global = len(df_global)
    for cfg in VALID_STATES:
        cnt = (df_global['Configuration'] == cfg).sum()
        rows_config.append({'Scope': 'Global', 'Configuration': cfg, 'Count': cnt, 'Percent': cnt / total_global * 100})
    for sec, df_sec in sector_dfs.items():
        total_sec = len(df_sec)
        for cfg in VALID_STATES:
            cnt = (df_sec['Configuration'] == cfg).sum()
            rows_config.append({'Scope': f'Sector_{sec}', 'Configuration': cfg, 'Count': cnt, 'Percent': cnt / total_sec * 100})
    pd.DataFrame(rows_config).to_csv(CSV_CONFIG, index=False, encoding='utf-8-sig')
    
    # 2. Descriptive statistics
    rows_stats = []
    for scope, df_tmp in [('Global', df_global)] + [(f'Sector_{s}', df_sec) for s, df_sec in sector_dfs.items()]:
        for var in NUMERIC_VARS:
            if var not in df_tmp.columns:
                continue
            for cfg in VALID_STATES:
                vals = df_tmp[df_tmp['Configuration'] == cfg][var].dropna()
                if len(vals) == 0:
                    continue
                rows_stats.append({
                    'Scope': scope, 'Configuration': cfg, 'Variable': var,
                    'N': len(vals), 'Mean': vals.mean(), 'Std': vals.std(),
                    'Min': vals.min(), 'Q1': vals.quantile(0.25),
                    'Median': vals.median(), 'Q3': vals.quantile(0.75), 'Max': vals.max()
                })
    pd.DataFrame(rows_stats).to_csv(CSV_STATS, index=False, encoding='utf-8-sig')
    
    # 3. Pairwise Mann–Whitney
    if all_pairwise:
        pd.concat(all_pairwise, ignore_index=True).to_csv(CSV_PAIRWISE, index=False, encoding='utf-8-sig')
    
    # 4. Markov matrices
    markov_rows = []
    markov_global['Scope'] = 'Global'
    markov_rows.append(markov_global)
    for sec, df_mark in markov_sectors.items():
        df_mark['Scope'] = f'Sector_{sec}'
        markov_rows.append(df_mark)
    if markov_rows:
        pd.concat(markov_rows, ignore_index=True).to_csv(CSV_MARKOV, index=False, encoding='utf-8-sig')
    
    # 5. Multivariate results (classification)
    multivar_rows = []
    # Global
    if class_global:
        acc, report, cm, importance = class_global
        for lab, metrics in report.items():
            if lab in ['accuracy', 'macro avg', 'weighted avg']:
                continue
            multivar_rows.append({
                'Scope': 'Global', 'Configuration': lab,
                'Precision': metrics['precision'], 'Recall': metrics['recall'],
                'F1_score': metrics['f1-score'], 'Support': metrics['support']
            })
        multivar_rows.append({
            'Scope': 'Global', 'Configuration': 'Accuracy',
            'Precision': np.nan, 'Recall': np.nan, 'F1_score': acc, 'Support': np.nan
        })
        for feat, imp in importance.items():
            multivar_rows.append({
                'Scope': 'Global', 'Configuration': f'Feature_{feat}',
                'Precision': imp, 'Recall': np.nan, 'F1_score': np.nan, 'Support': np.nan
            })
    # Sector
    for sec, mv in class_sectors.items():
        if mv:
            acc, report, cm, importance = mv
            for lab, metrics in report.items():
                if lab in ['accuracy', 'macro avg', 'weighted avg']:
                    continue
                multivar_rows.append({
                    'Scope': f'Sector_{sec}', 'Configuration': lab,
                    'Precision': metrics['precision'], 'Recall': metrics['recall'],
                    'F1_score': metrics['f1-score'], 'Support': metrics['support']
                })
            multivar_rows.append({
                'Scope': f'Sector_{sec}', 'Configuration': 'Accuracy',
                'Precision': np.nan, 'Recall': np.nan, 'F1_score': acc, 'Support': np.nan
            })
    if multivar_rows:
        pd.DataFrame(multivar_rows).to_csv(CSV_MULTIVAR, index=False, encoding='utf-8-sig')
    
    print(f"All CSV files exported to {OUTPUT_DIR}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 80)
    print("BASIC DESCRIPTIVES – ACADEMIC ANALYSIS")
    print("Global and Sector-Level Statistical Summary")
    print("=" * 80)
    
    # Load data
    df = load_and_prepare_data()
    
    # Global
    df_global = df.copy()
    
    # Sectors with at least 30 observations
    sector_dfs = {}
    if 'Sector' in df.columns:
        sectors = df['Sector'].dropna().unique()
        for sec in sectors:
            df_sec = df[df['Sector'] == sec].copy()
            if len(df_sec) >= 30:
                sector_dfs[sec] = df_sec
        print(f"Found {len(sector_dfs)} sectors with ≥30 observations.")
    else:
        print("No 'Sector' column found – sector breakdown skipped.")
    
    # Pairwise Mann–Whitney (collect for all scopes)
    all_pairwise = []
    # Global
    pairwise_by_var_global = {}
    for var in NUMERIC_VARS:
        if var in df_global.columns:
            df_pair = pairwise_mannwhitney(df_global, var)
            if not df_pair.empty:
                df_pair['Variable'] = var
                df_pair['Scope'] = 'Global'
                all_pairwise.append(df_pair)
                pairwise_by_var_global[var] = df_pair
    # Sectors
    for sec, df_sec in sector_dfs.items():
        for var in NUMERIC_VARS:
            if var in df_sec.columns:
                df_pair = pairwise_mannwhitney(df_sec, var)
                if not df_pair.empty:
                    df_pair['Variable'] = var
                    df_pair['Scope'] = f'Sector_{sec}'
                    all_pairwise.append(df_pair)
    
    # Markov matrices
    group_col = 'Cycle_ID' if 'Cycle_ID' in df.columns else 'Ticker'
    markov_global = markov_transition_matrix(df_global, group_col, VALID_STATES)
    markov_sectors = {}
    for sec, df_sec in sector_dfs.items():
        markov_sectors[sec] = markov_transition_matrix(df_sec, group_col, VALID_STATES)
    
    # Multivariate: MANOVA and Decision Tree
    manova_global = manova_wilks(df_global, ['E_3', 'R_t', 'PDI_t'])
    class_global = decision_tree_classification(df_global, ['E_3', 'R_t', 'PDI_t'], 'Configuration')
    class_sectors = {}
    for sec, df_sec in sector_dfs.items():
        class_sectors[sec] = decision_tree_classification(df_sec, ['E_3', 'R_t', 'PDI_t'], 'Configuration')
    
    # Write academic report
    with open(TXT_REPORT, 'w', encoding='utf-8') as f:
        write_academic_report(df_global, sector_dfs, pairwise_by_var_global, markov_global,
                              manova_global, class_global, f)
    
    # Export all CSVs
    export_all_csv(df_global, sector_dfs, all_pairwise, markov_global, markov_sectors,
                   class_global, class_sectors)
    
    print(f"\n✅ Analysis completed successfully.")
    print(f"   Academic report: {TXT_REPORT}")
    print(f"   All CSV outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
