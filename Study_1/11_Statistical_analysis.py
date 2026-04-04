"""
STEP 11: STRUCTURAL VALIDATION REPORT
Objective: Executes empirical validation of the theoretical hypothesis using longitudinal panel data.
Core Analyses:
1. Quarter-over-quarter Markov transition matrices with cluster bootstrap confidence intervals.
2. Gestation (C2) pathway analysis determining empirical E3 thresholds predicting maturation vs. collapse.
3. Leading indicator validation for Productive Discharge Index (PDI) anticipating structural collapse.
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
DATA_FILE = 'data/final_all_cycles_combined.csv'   
OUTPUT_FILE = 'structural_report_complete.txt'
VALID_STATES = ['Normal', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
N_BOOT = 1000
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ==================== DATA ACQUISITION & SORTING ====================
print("Loading longitudinal panel data...")
df = pd.read_csv(DATA_FILE)
df['period_end'] = pd.to_datetime(df['period_end'])

# Standardize configuration terminology
if 'Regime_Label' in df.columns:
    df['Configuration'] = np.where(df['Regime_Label'] == 'Normal_Regime', 'Normal', df['Configuration'])

# Strict temporal sorting to ensure quarter-over-quarter transition validity
df = df.sort_values(['Ticker', 'period_end']).reset_index(drop=True)

# Identify hierarchical grouping structure for time-series isolation
group_col = 'Cycle_ID' if 'Cycle_ID' in df.columns else 'Ticker'

# ==================== 1. QUARTERLY MARKOV TRANSITION MATRIX ====================
print("Computing quarter-over-quarter Markov transition matrix with bootstrap CI...")

if group_col == 'Cycle_ID':
    sequences = df.groupby('Cycle_ID')['Configuration'].agg(list).tolist()
else:
    sequences = df.groupby('Ticker')['Configuration'].agg(list).tolist()
sequences = [seq for seq in sequences if len(seq) >= 2]

def compute_transition_probs(seqs):
    """Calculates empirical probabilities for state transitions between consecutive quarters."""
    counts = {s: {t:0 for t in VALID_STATES} for s in VALID_STATES}
    total = {s:0 for s in VALID_STATES}
    for seq in seqs:
        for i in range(len(seq)-1):
            s, t = seq[i], seq[i+1] # Evaluates adjacent quarters (t and t+1)
            if s in counts and t in counts[s]:
                counts[s][t] += 1
                total[s] += 1
    probs = {s: {t: counts[s][t]/total[s] if total[s]>0 else 0 for t in VALID_STATES} for s in VALID_STATES}
    return probs, counts, total

orig_probs, orig_counts, orig_total = compute_transition_probs(sequences)

# Cluster bootstrap resampling for robust confidence intervals
boot_probs = []
for _ in range(N_BOOT):
    boot_seqs = [sequences[i] for i in np.random.choice(len(sequences), len(sequences), replace=True)]
    p, _, _ = compute_transition_probs(boot_seqs)
    boot_probs.append(p)

def get_ci(probs_list, s, t):
    vals = [p[s][t] for p in probs_list]
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)

# ==================== 2. GESTATION (C2) BIFURCATION ANALYSIS ====================
print("Analyzing C2 threshold pathways...")
c2_cycles = []
for gid, group in df[df['Configuration']=='C2'].groupby(group_col):
    group = group.sort_values('period_end')
    if len(group) == 0: continue
    last = group.iloc[-1]
    
    next_idx = group.index[-1] + 1
    next_state = None
    if next_idx < len(df) and df.loc[next_idx, group_col] == gid:
        next_state = df.loc[next_idx, 'Configuration']
        
    if next_state in ['C3', 'C1', 'C6']:
        c2_cycles.append({'E3_end': last['E_3'], 'next_state': next_state})

c2_df = pd.DataFrame(c2_cycles)
to_c3 = c2_df[c2_df['next_state']=='C3']['E3_end'].dropna()
to_crash = c2_df[c2_df['next_state'].isin(['C1','C6'])]['E3_end'].dropna()

# Non-parametric assessment of systemic risk thresholds
if len(to_c3)>=3 and len(to_crash)>=3:
    u_e3, p_e3 = stats.mannwhitneyu(to_c3, to_crash, alternative='two-sided')
else:
    p_e3 = np.nan

# Empirical optimization of the E3 theoretical boundary
best_acc, best_th = 0, 10
if len(to_c3)>0 and len(to_crash)>0:
    for th in range(5, 51):
        correct_c3 = (to_c3 < th).sum()
        correct_crash = (to_crash >= th).sum()
        acc = (correct_c3 + correct_crash) / (len(to_c3)+len(to_crash))
        if acc > best_acc: best_acc, best_th = acc, th

# ==================== 3. PDI LEADING INDICATOR ANALYSIS ====================
print("Analyzing PDI as a forward-looking indicator within C3/C4...")
# Lag method: Classify current quarter based on the subsequent quarter's structural state
df['is_Crash'] = df['Configuration'].isin(['C1','C6']).astype(int)
df['Crash_next'] = df.groupby(group_col)['is_Crash'].shift(-1)

df_c3c4 = df[df['Configuration'].isin(['C3','C4'])].copy()
crash_pdi = df_c3c4[df_c3c4['Crash_next']==1]['PDI_t'].dropna()
safe_pdi = df_c3c4[df_c3c4['Crash_next']==0]['PDI_t'].dropna()

if len(crash_pdi)>=3 and len(safe_pdi)>=3:
    u_pdi, p_pdi = stats.mannwhitneyu(crash_pdi, safe_pdi, alternative='two-sided')
else:
    p_pdi = np.nan

# ==================== 4. ACADEMIC REPORT GENERATION ====================
print("Generating comprehensive statistical output...")
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    f.write("COMPLETE STRUCTURAL VALIDATION REPORT - CHIMERA HYPOTHESIS\n")
    f.write("="*80 + "\n\n")
    
    # 1. Distribution summary
    f.write("1. QUARTERLY CONFIGURATION DISTRIBUTION\n")
    f.write("-"*40 + "\n")
    config_counts = df['Configuration'].value_counts()
    total = len(df)
    for c in VALID_STATES:
        cnt = config_counts.get(c, 0)
        f.write(f"  {c:6} : {cnt:5} ({cnt/total*100:5.2f}%)\n" if total>0 else "")
    
    # 2. Markov Matrix Output
    f.write("\n2. QUARTER-OVER-QUARTER MARKOV TRANSITION MATRIX (95% CI)\n")
    f.write("-"*60 + "\n")
    header = "From\\To    " + " ".join(f"{s:>6}" for s in VALID_STATES)
    f.write(header + "\n")
    for s in VALID_STATES:
        row = f"{s:6}    "
        for t in VALID_STATES:
            orig = orig_probs[s][t] * 100
            if orig_total.get(s, 0) >= 5:
                low, high = get_ci(boot_probs, s, t)
                row += f"{orig:6.1f}[{low*100:4.0f}-{high*100:4.0f}] "
            else:
                row += f"{orig:6.1f}         "
        f.write(row + "\n")
    
    # 3. C2 Path Analysis
    f.write("\n3. GESTATION (C2) PATHWAY DYNAMICS\n")
    f.write("-"*50 + "\n")
    f.write(f"C2 observations mapped to immediate subsequent state: {len(c2_df)}\n")
    f.write(f"  -> Maturation (C2→C3) : n={len(to_c3)}, Median E3={to_c3.median():.2f}\n")
    f.write(f"  -> Collapse (C2→C1/C6): n={len(to_crash)}, Median E3={to_crash.median():.2f}\n")
    if not np.isnan(p_e3): f.write(f"Mann-Whitney U: p={p_e3:.4e}\n")
    f.write(f"Empirically optimal E3 barrier: {best_th} (Accuracy={best_acc:.1%})\n")
    
    # 4. PDI Forecasting
    f.write("\n4. PDI FORECASTING UTILITY WITHIN MATURITY (C3/C4)\n")
    f.write("-"*50 + "\n")
    f.write(f"  -> Imminent Collapse Next Quarter: n={len(crash_pdi)}, Median PDI={crash_pdi.median():.4f}\n")
    f.write(f"  -> Structural Continuity Next Quarter: n={len(safe_pdi)}, Median PDI={safe_pdi.median():.4f}\n")
    if not np.isnan(p_pdi):
        f.write(f"Mann-Whitney U: p={p_pdi:.4e}\n")
        f.write("  => Validated Leading Indicator\n" if p_pdi < 0.05 else "  => No statistical forecasting divergence\n")
    
    # 5. Cross-sectional Rank Correlations
    f.write("\n5. CROSS-SECTIONAL NON-PARAMETRIC CORRELATIONS\n")
    f.write("-"*50 + "\n")
    rho_rt, p_rt = stats.spearmanr(df['E_3'], df['R_t'], nan_policy='omit')
    rho_pdi, p_pdi = stats.spearmanr(df['E_3'], df['PDI_t'], nan_policy='omit')
    f.write(f"  E3 vs Discharge Velocity (R_t) : rho = {rho_rt:.4f}, p = {p_rt:.2e}\n")
    f.write(f"  E3 vs Discharge Source (PDI_t) : rho = {rho_pdi:.4f}, p = {p_pdi:.2e}\n")

print(f"Report complete. File saved to {OUTPUT_FILE}")