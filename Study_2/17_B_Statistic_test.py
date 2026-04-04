import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve

df = pd.read_csv('data/final_panel.csv')
df['period_end'] = pd.to_datetime(df['period_end'])
if 'Regime_Label' in df.columns:
    df['Configuration'] = np.where(df['Regime_Label'] == 'Normal_Regime', 'Normal', df['Configuration'])

df = df.sort_values(['Ticker', 'period_end']).reset_index(drop=True)
group_col = 'Cycle_ID' if 'Cycle_ID' in df.columns else 'Ticker'

# Tính PGR_t nếu chưa có
if 'PGR_t' not in df.columns:
    df['PGR_t'] = df.groupby('Ticker')['V_Prod_base'].pct_change()

# Tính B
df['A'] = 1 + df['PGR_t']
df['B'] = df['E_3'] - df['A']

# Lấy các chuỗi C2
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
        c2_cycles.append({
            'E3_end': last['E_3'],
            'B_end': last['B'],
            'next_state': next_state
        })
c2_df = pd.DataFrame(c2_cycles).dropna()
y = (c2_df['next_state'].isin(['C1','C6'])).astype(int)

# ROC cho E3
fpr_e3, tpr_e3, th_e3 = roc_curve(y, c2_df['E3_end'])
youden_e3 = np.argmax(tpr_e3 - fpr_e3)
acc_e3 = (tpr_e3[youden_e3] + 1 - fpr_e3[youden_e3]) / 2
th_opt_e3 = th_e3[youden_e3]

# ROC cho B
fpr_b, tpr_b, th_b = roc_curve(y, c2_df['B_end'])
youden_b = np.argmax(tpr_b - fpr_b)
acc_b = (tpr_b[youden_b] + 1 - fpr_b[youden_b]) / 2
th_opt_b = th_b[youden_b]

print("=== SO SÁNH DỰ BÁO C2→C3 vs C2→Crash ===")
print(f"E3: threshold = {th_opt_e3:.2f}, accuracy = {acc_e3:.1%}")
print(f"B : threshold = {th_opt_b:.2f}, accuracy = {acc_b:.1%}")
if acc_b > acc_e3:
    print("=> B có accuracy cao hơn E3 (better predictor).")
elif acc_b < acc_e3:
    print("=> E3 có accuracy cao hơn B.")
else:
    print("=> B và E3 có accuracy tương đương.")