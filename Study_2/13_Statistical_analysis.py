"""
FINAL STRUCTURAL VALIDATION REPORT
- Markov matrix with bootstrap confidence intervals
- C2 path: E3 threshold analysis (C2->C3 vs C2->C1/C6)
- PDI leading indicator within C3/C4 (using next-quarter crash)
- Summary statistics
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ========== CONFIG ==========
DATA_FILE = 'data/final_panel.csv'   # Đường dẫn đến file dữ liệu
OUTPUT_FILE = 'structural_report_complete.txt'
VALID_STATES = ['Normal', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
N_BOOT = 1000
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ========== LOAD DATA ==========
print("Loading data...")
df = pd.read_csv(DATA_FILE)
df['period_end'] = pd.to_datetime(df['period_end'])
if 'Regime_Label' in df.columns:
    df['Configuration'] = np.where(df['Regime_Label'] == 'Normal_Regime', 'Normal', df['Configuration'])

# Sort by company and time
df = df.sort_values(['Ticker', 'period_end']).reset_index(drop=True)

# Identify grouping column
group_col = 'Cycle_ID' if 'Cycle_ID' in df.columns else 'Ticker'

# ========== 1. MARKOV MATRIX WITH BOOTSTRAP CI ==========
print("Computing Markov transition matrix with bootstrap CI...")
# Get state sequences per cycle
if group_col == 'Cycle_ID':
    sequences = df.groupby('Cycle_ID')['Configuration'].agg(list).tolist()
else:
    sequences = df.groupby('Ticker')['Configuration'].agg(list).tolist()
sequences = [seq for seq in sequences if len(seq) >= 2]

def compute_transition_probs(seqs):
    counts = {s: {t:0 for t in VALID_STATES} for s in VALID_STATES}
    total = {s:0 for s in VALID_STATES}
    for seq in seqs:
        for i in range(len(seq)-1):
            s, t = seq[i], seq[i+1]
            if s in counts and t in counts[s]:
                counts[s][t] += 1
                total[s] += 1
    probs = {s: {t: counts[s][t]/total[s] if total[s]>0 else 0 for t in VALID_STATES} for s in VALID_STATES}
    return probs, counts, total

orig_probs, orig_counts, orig_total = compute_transition_probs(sequences)

# Bootstrap
boot_probs = []
for _ in range(N_BOOT):
    boot_seqs = [sequences[i] for i in np.random.choice(len(sequences), len(sequences), replace=True)]
    p, _, _ = compute_transition_probs(boot_seqs)
    boot_probs.append(p)

def get_ci(probs_list, s, t):
    vals = [p[s][t] for p in probs_list]
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)

# ========== 2. C2 PATH: E3 THRESHOLD ==========
print("Analyzing C2 paths...")
c2_cycles = []
for gid, group in df[df['Configuration']=='C2'].groupby(group_col):
    group = group.sort_values('period_end')
    if len(group) == 0: continue
    last = group.iloc[-1]
    # Find next state
    next_idx = group.index[-1] + 1
    next_state = None
    if next_idx < len(df) and df.loc[next_idx, group_col] == gid:
        next_state = df.loc[next_idx, 'Configuration']
    if next_state in ['C3', 'C1', 'C6']:
        c2_cycles.append({
            'E3_end': last['E_3'],
            'next_state': next_state
        })
c2_df = pd.DataFrame(c2_cycles)
to_c3 = c2_df[c2_df['next_state']=='C3']['E3_end'].dropna()
to_crash = c2_df[c2_df['next_state'].isin(['C1','C6'])]['E3_end'].dropna()

# Mann-Whitney
if len(to_c3)>=3 and len(to_crash)>=3:
    u_e3, p_e3 = stats.mannwhitneyu(to_c3, to_crash, alternative='two-sided')
else:
    p_e3 = np.nan

# Optimal threshold
best_acc = 0
best_th = 10
if len(to_c3)>0 and len(to_crash)>0:
    for th in range(5, 51):
        correct_c3 = (to_c3 < th).sum()
        correct_crash = (to_crash >= th).sum()
        acc = (correct_c3 + correct_crash) / (len(to_c3)+len(to_crash))
        if acc > best_acc:
            best_acc = acc
            best_th = th

# ========== 3. PDI LEADING INDICATOR IN C3/C4 ==========
print("Analyzing PDI in C3/C4...")
# Create next crash indicator (lag method)
df['is_Crash'] = df['Configuration'].isin(['C1','C6']).astype(int)
df['Crash_next'] = df.groupby(group_col)['is_Crash'].shift(-1)

df_c3c4 = df[df['Configuration'].isin(['C3','C4'])].copy()
crash_pdi = df_c3c4[df_c3c4['Crash_next']==1]['PDI_t'].dropna()
safe_pdi = df_c3c4[df_c3c4['Crash_next']==0]['PDI_t'].dropna()
if len(crash_pdi)>=3 and len(safe_pdi)>=3:
    u_pdi, p_pdi = stats.mannwhitneyu(crash_pdi, safe_pdi, alternative='two-sided')
else:
    p_pdi = np.nan

# ========== 4. WRITE REPORT ==========
print("Writing report...")
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    f.write("COMPLETE STRUCTURAL VALIDATION REPORT - CHIMERA HYPOTHESIS\n")
    f.write("="*80 + "\n\n")
    
    # Configuration distribution
    f.write("1. CONFIGURATION DISTRIBUTION\n")
    f.write("-"*40 + "\n")
    config_counts = df['Configuration'].value_counts()
    total = len(df)
    for c in VALID_STATES:
        cnt = config_counts.get(c, 0)
        pct = cnt/total*100
        f.write(f"  {c:6} : {cnt:5} ({pct:5.2f}%)\n")
    f.write("\n")
    
    # Markov matrix with CI
    f.write("2. MARKOV TRANSITION MATRIX (with 95% bootstrap CI)\n")
    f.write("-"*60 + "\n")
    f.write("Transition probabilities (%), with 95% CI from bootstrap (by cycle):\n\n")
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
    f.write("\nNote: CI shown only for states with at least 5 outgoing transitions.\n\n")
    
    # C2 path
    f.write("3. C2 PATH ANALYSIS: E3 AS THRESHOLD\n")
    f.write("-"*50 + "\n")
    f.write(f"Number of C2 sequences with known next state: {len(c2_df)}\n")
    f.write(f"  -> C2→C3 : {len(to_c3)} (median E3 = {to_c3.median():.2f})\n")
    f.write(f"  -> C2→C1/C6: {len(to_crash)} (median E3 = {to_crash.median():.2f})\n")
    if not np.isnan(p_e3):
        f.write(f"Mann-Whitney U test: p = {p_e3:.4e} (significant difference)\n")
    f.write(f"Optimal E3 threshold (max accuracy): {best_th} (accuracy = {best_acc:.1%})\n\n")
    f.write("Accuracy at various thresholds:\n")
    for th in [5,10,15,20,25,30,35,40]:
        correct_c3 = (to_c3 < th).sum()
        correct_crash = (to_crash >= th).sum()
        acc = (correct_c3 + correct_crash) / (len(to_c3)+len(to_crash))
        f.write(f"  threshold={th:2}: correct C3={correct_c3:2}/{len(to_c3):2}, correct crash={correct_crash:2}/{len(to_crash):2}, total accuracy={acc:.1%}\n")
    f.write("\n")
    
    # PDI in C3/C4
    f.write("4. PDI LEADING INDICATOR WITHIN C3/C4\n")
    f.write("-"*50 + "\n")
    f.write(f"Number of C3/C4 quarters with known next quarter: {len(df_c3c4)}\n")
    f.write(f"  -> Next crash (C1/C6): n={len(crash_pdi)}, median PDI={crash_pdi.median():.4f}\n")
    f.write(f"  -> Next safe (others): n={len(safe_pdi)}, median PDI={safe_pdi.median():.4f}\n")
    if not np.isnan(p_pdi):
        f.write(f"Mann-Whitney U test: p = {p_pdi:.4e}\n")
        if p_pdi < 0.05:
            f.write("  => PDI is significantly lower before crash (leading indicator)\n")
        else:
            f.write("  => No significant difference\n")
    else:
        f.write("  Insufficient data for test (need at least 3 crash quarters).\n")
    f.write("\n")
    
    # Additional stats
    f.write("5. ADDITIONAL STATISTICS\n")
    f.write("-"*30 + "\n")
    f.write("Median E3 by configuration:\n")
    for c in VALID_STATES:
        med = df[df['Configuration']==c]['E_3'].median()
        f.write(f"  {c:6}: {med:.4f}\n")
    f.write("\nMedian PDI by configuration:\n")
    for c in VALID_STATES:
        med = df[df['Configuration']==c]['PDI_t'].median()
        f.write(f"  {c:6}: {med:.4f}\n")
    f.write("\nSpearman correlations (cross-sectional):\n")
    rho_rt, p_rt = stats.spearmanr(df['E_3'], df['R_t'], nan_policy='omit')
    rho_pdi, p_pdi = stats.spearmanr(df['E_3'], df['PDI_t'], nan_policy='omit')
    f.write(f"  E3 vs R_t   : rho = {rho_rt:.4f}, p = {p_rt:.2e}\n")
    f.write(f"  E3 vs PDI_t : rho = {rho_pdi:.4f}, p = {p_pdi:.2e}\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("END OF REPORT\n")

print(f"Report saved to {OUTPUT_FILE}")