import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# 1. HÀM PHÂN LOẠI (Tích hợp từ file 11 của bạn)
# ==========================================
def classify_state(row):
    """Logic phân loại C1-C6 chính xác từ file của bạn"""
    rt = row.get('R_t')
    dk = row.get('dK_Pi_prime')
    s = row.get('s_total')
    dkp = row.get('dK_Pi_prime_pct')
    speculative = row.get('Speculative_Regime')

    if pd.isna(rt) or pd.isna(dk):
        return 'N/A'
    if not speculative:
        return 'Normal'

    rt_tol = 1e-6

    if rt <= rt_tol and dk < 0 and s <= 0:
        return 'C1' if dkp <= -0.15 else 'C6'
    if rt <= rt_tol and dk > 0 and s <= 0:
        return 'C2'
    if rt > rt_tol and dk > 0 and s > 0:
        return 'C3'
    if rt_tol < rt < 0.999 and dk < 0 and s > 0:
        return 'C4'
    if rt >= 0.999 and dk < 0 and s > 0:
        return 'C5'

    return 'Other'

def apply_classification(df):
    """Hàm wrapper để chạy toàn bộ DataFrame"""
    df_out = df.copy()
    
    # Xử lý dị thường vô cực
    core_cols = ['E_3', 'R_t', 'PDI_t', 'PGR_t']
    for col in core_cols:
        if col in df_out.columns:
            df_out[col] = df_out[col].replace([np.inf, -np.inf], np.nan)
            
    # Xử lý R_t an toàn
    if 'R_t' in df_out.columns and 's_total' in df_out.columns:
        df_out['R_t'] = np.where(df_out['s_total'] <= 0, 0.0, df_out['R_t'])

    # Phân loại Raw
    df_out['Raw_Configuration'] = df_out.apply(classify_state, axis=1)

    # Gác cổng (Formal Gate)
    if 'Speculative_Regime' in df_out.columns:
        df_out['Configuration'] = np.where(
            (df_out['Speculative_Regime'] == False) & (df_out['Raw_Configuration'] != 'N/A'), 
            'Normal', 
            df_out['Raw_Configuration']
        )
    return df_out

# ==========================================
# 2. THIẾT LẬP STRESS TEST
# ==========================================
print("="*75)
print(" KBRAND ROBUSTNESS TEST (SENSITIVITY ANALYSIS)")
print("="*75)

try:
    df_base = pd.read_csv('data/final_panel.csv')
    df_base['period_end'] = pd.to_datetime(df_base['period_end'])
    df_base = df_base.sort_values(['Ticker', 'period_end']).reset_index(drop=True)
except FileNotFoundError:
    print("❌ Lỗi: Không tìm thấy file 'final_panel.csv'. Vui lòng kiểm tra lại đường dẫn.")
    exit()

group_col = 'Cycle_ID' if 'Cycle_ID' in df_base.columns else 'Ticker'

# Chạy phân loại Baseline (Gốc)
df_base = apply_classification(df_base)
baseline_configs = df_base['Configuration'].copy()

# Kiểm tra cột KBrand
if 'KBrand' not in df_base.columns:
    print("❌ Lỗi: Không tìm thấy cột 'KBrand' trong dữ liệu. Stress test không thể thực hiện.")
    exit()

# Các kịch bản nhân KBrand (Giảm 50% đến Tăng 50%)
multipliers = [0.5, 0.8, 0.9, 1.1, 1.2, 1.5]
results = []

# ==========================================
# 3. VÒNG LẶP TEST KỊCH BẢN
# ==========================================
for m in multipliers:
    df_test = df_base.copy()
    
    # 3.1 Điều chỉnh KBrand và K_Pi_prime
    # Khi KBrand thay đổi, KPi' (vốn là phần dư) sẽ biến động bù trừ
    KBrand_old = df_test['KBrand']
    KBrand_new = KBrand_old * m
    
    if 'K_Pi_prime' in df_test.columns:
        df_test['K_Pi_prime'] = df_test['K_Pi_prime'] + (KBrand_old - KBrand_new)
    else:
        print("❌ Lỗi: Thiếu cột 'K_Pi_prime'.")
        break
        
    # 3.2 Tính lại độ trễ và động lượng của K_Pi_prime
    df_test['K_Pi_prime_lag1'] = df_test.groupby(group_col)['K_Pi_prime'].shift(1)
    df_test['dK_Pi_prime'] = df_test['K_Pi_prime'] - df_test['K_Pi_prime_lag1']
    
    # Tính phần trăm thay đổi (tránh chia cho 0)
    df_test['dK_Pi_prime_pct'] = np.where(
        df_test['K_Pi_prime_lag1'].abs() > 0,
        df_test['dK_Pi_prime'] / df_test['K_Pi_prime_lag1'].abs(),
        np.nan
    )
    
    # 3.3 Tính lại E_3 và R_t
    if 'V_Prod_base' in df_test.columns:
        df_test['E_3'] = df_test['K_Pi_prime'] / df_test['V_Prod_base']
        
    df_test['R_t'] = np.where(
        df_test['K_Pi_prime_lag1'] > 0, 
        df_test['s_total'] / df_test['K_Pi_prime_lag1'], 
        0.0
    )
    
    # 3.4 Cập nhật Gate Đầu cơ (Speculative Regime)
    # Theo lý thuyết: E_3 > 1 và R_t < 1
    df_test['Speculative_Regime'] = (df_test['E_3'] > 1) & (df_test['R_t'] < 1)
    
    # 3.5 Chạy lại hàm phân loại của bạn
    df_test = apply_classification(df_test)
    
    # 3.6 Đo lường độ vững (Robustness)
    matches = (df_test['Configuration'] == baseline_configs)
    
    # Chỉ tính tỷ lệ ổn định trên những dòng KHÔNG phải là N/A hoặc Normal ngay từ đầu
    # (Để biết chính xác cấu trúc đầu cơ có bị sụp đổ thành Normal hay chuyển pha không)
    valid_mask = ~baseline_configs.isin(['N/A', 'Unknown'])
    stability_rate = matches[valid_mask].mean() * 100
    changed_count = (~matches[valid_mask]).sum()
    
    results.append({
        'KBrand_Multiplier': f"{m}x ({(m-1)*100:+.0f}%)",
        'Stability_Rate': f"{stability_rate:.2f}%",
        'Changed_Rows': changed_count
    })
    
    print(f"✅ Hoàn thành kịch bản KBrand thay đổi {(m-1)*100:+.0f}%")

# ==========================================
# 4. XUẤT KẾT QUẢ
# ==========================================
results_df = pd.DataFrame(results)

print("\n" + "="*75)
print(f"{'Biến thiên KBrand':<25} | {'Tỷ lệ Ổn định (Stability)':<25} | {'Số dòng thay đổi nhãn'}")
print("-" * 75)
for idx, row in results_df.iterrows():
    print(f"{row['KBrand_Multiplier']:<25} | {row['Stability_Rate']:<25} | {row['Changed_Rows']}")
print("="*75)

results_df.to_csv('KBrand_Robustness_Report.csv', index=False)
print("\nĐã xuất kết quả ra file 'KBrand_Robustness_Report.csv'.")
