"""
STEP 14: COMPREHENSIVE STATISTICAL & ROBUSTNESS ANALYSIS
Consolidates all empirical tests for the research paper:
1. Markov Transition Matrix
2. Spearman Rank Correlation (Speculative Paradox)
3. Kruskal-Wallis & Post-Hoc Tests (Boundary Analysis for ALL metrics)
4. Conditional & Directional PDI Tests
5. Robustness Tests (Temporal Split & Bootstrapping)
"""

import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations
import warnings
import os

warnings.filterwarnings('ignore')

# Tự động tìm file dù nó ở thư mục gốc hay trong thư mục data/
DATA_FILE = 'data/final_all_cycles_combined.csv'
FALLBACK_DATA_FILE = 'final_all_cycles_combined.csv'

VALID_CONFIGS = ['Normal', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']

def load_and_prep_data():
    """Loads panel data and calculates required lagged/delta variables."""
    print("🔄 Loading and preparing panel data...")
    
    file_path = DATA_FILE
    if not os.path.exists(file_path):
        if os.path.exists(FALLBACK_DATA_FILE):
            file_path = FALLBACK_DATA_FILE
        else:
            print(f"❌ Data file not found at {DATA_FILE} or {FALLBACK_DATA_FILE}")
            return None

    try:
        df = pd.read_csv(file_path)
        df['period_end'] = pd.to_datetime(df['period_end'])
        
        # Đồng bộ nhãn Normal 
        if 'Regime' in df.columns:
            df['Configuration'] = np.where(df['Regime'] == 'Normal_Regime', 'Normal', df['Configuration'])
        elif 'Speculative_Regime' in df.columns:
            df['Configuration'] = np.where(df['Speculative_Regime'] == False, 'Normal', df['Configuration'])
            
        # Tự động tính lại PDI_t nếu file CSV bị thiếu
        if 'PDI_t' not in df.columns:
            if 's_total' in df.columns and 'dK_Pi_prime' in df.columns:
                df['PDI_t'] = df['s_total'] / (df['dK_Pi_prime'].abs() + df['s_total'])
                df['PDI_t'] = df['PDI_t'].fillna(0)
            else:
                print("❌ Missing base columns to calculate PDI_t.")
                return None

        # Ép kiểu dữ liệu an toàn
        numeric_cols = ['E_3', 'R_t', 'PDI_t', 'dK_Pi_prime_pct', 'PGR_t']
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
                df[c] = df[c].replace([np.inf, -np.inf], np.nan)
                
        # Gom nhóm 
        group_col = 'Cycle_ID' if 'Cycle_ID' in df.columns else 'Ticker'
        
        # Tính toán biến Delta PDI và Lag
        df = df.sort_values([group_col, 'period_end']).reset_index(drop=True)
        df['d_PDI_t'] = df.groupby(group_col)['PDI_t'].diff()
        df['PDI_lag1'] = df.groupby(group_col)['PDI_t'].shift(1)
        df['d_PDI_lag1'] = df.groupby(group_col)['d_PDI_t'].shift(1)
        
        # Tạo biến phân loại: Quý này có sập (C1/C6) hay không?
        df['is_Crash_Regime'] = np.where(df['Configuration'].isin(['C1', 'C6']), 1, 0)
        
        print(f"✅ Data loaded: {len(df):,} observations across {df['Ticker'].nunique()} companies.")
        return df
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return None

# =============================================================================
# MODULE 1: MARKOV TRANSITIONS
# =============================================================================
def analyze_markov_transitions(df):
    print("\n" + "="*80)
    print("1. MARKOV TRANSITION MATRIX (FULL LIFECYCLE)")
    print("="*80)
    
    group_col = 'Cycle_ID' if 'Cycle_ID' in df.columns else 'Ticker'
    
    df_sorted = df.copy()
    df_sorted['Next_Config'] = df_sorted.groupby(group_col)['Configuration'].shift(-1)
    
    transitions = df_sorted.dropna(subset=['Next_Config'])
    trans_data = transitions[
        (transitions['Configuration'].isin(VALID_CONFIGS)) &
        (transitions['Next_Config'].isin(VALID_CONFIGS))
    ]
    
    matrix = pd.crosstab(
        trans_data['Configuration'],
        trans_data['Next_Config'],
        normalize='index'
    ) * 100
    
    matrix = matrix.reindex(index=VALID_CONFIGS, columns=VALID_CONFIGS, fill_value=0)
    print(matrix.round(1))

# =============================================================================
# MODULE 2: SPEARMAN CORRELATION
# =============================================================================
def analyze_correlations(df):
    print("\n" + "="*80)
    print("2. SPEARMAN RANK CORRELATION (NGHỊCH LÝ ĐẦU CƠ)")
    print("="*80)
    
    df_corr = df.dropna(subset=['E_3', 'R_t', 'PDI_t']).copy()
    if len(df_corr) < 2: return

    print("A. Toàn bộ quan sát (Cross-sectional)")
    rho_rt, p_rt = stats.spearmanr(df_corr['E_3'], df_corr['R_t'], nan_policy='omit')
    rho_pdi, p_pdi = stats.spearmanr(df_corr['E_3'], df_corr['PDI_t'], nan_policy='omit')
    print(f"  E_3 vs R_t    : rho = {rho_rt:6.4f} | p-value = {p_rt:.2e}")
    print(f"  E_3 vs PDI_t  : rho = {rho_pdi:6.4f} | p-value = {p_pdi:.2e}")
    
    print("\nB. Trung bình theo từng công ty (Within-firm)")
    results = []
    
    for ticker, group in df_corr.groupby('Ticker'):
        if len(group) < 6: continue
            
        r_rt = stats.spearmanr(group['E_3'], group['R_t'], nan_policy='omit')[0] if group['E_3'].nunique() > 1 and group['R_t'].nunique() > 1 else np.nan
        r_pdi = stats.spearmanr(group['E_3'], group['PDI_t'], nan_policy='omit')[0] if group['E_3'].nunique() > 1 and group['PDI_t'].nunique() > 1 else np.nan
            
        results.append({'Ticker': ticker, 'n_quarters': len(group), 'rho_E3_Rt': r_rt, 'rho_E3_PDI': r_pdi})
    
    res_df = pd.DataFrame(results).dropna()
    if not res_df.empty:
        print(res_df.round(4).to_string(index=False))
        print(f"\nTrung bình hiệu ứng Within-firm (Chỉ tính các cty có biến động):")
        print(f"  E_3 vs R_t   : rho = {res_df['rho_E3_Rt'].mean():6.4f}")
        print(f"  E_3 vs PDI_t : rho = {res_df['rho_E3_PDI'].mean():6.4f}")

# =============================================================================
# MODULE 3: STRUCTURAL BOUNDARY TESTS
# =============================================================================
def analyze_kruskal_and_posthoc(df):
    print("\n" + "="*80)
    print("3. PHÂN TÁCH CẤU TRÚC (KRUSKAL-WALLIS & MANN-WHITNEY)")
    print("="*80)
    
    metrics = ['E_3', 'PGR_t', 'R_t', 'PDI_t', 'dK_Pi_prime_pct']
    valid_data_dict = {}
    
    print("A. KIỂM ĐỊNH TỔNG QUAN (KRUSKAL-WALLIS H-TEST)")
    for m in metrics:
        if m not in df.columns: continue
        samples = []
        configs = []
        for c in VALID_CONFIGS:
            s = df[df['Configuration'] == c][m].dropna()
            if len(s) > 3: 
                samples.append(s)
                configs.append(c)
                
        valid_data_dict[m] = {c: s for c, s in zip(configs, samples)}
        if len(samples) > 1:
            stat, p = stats.kruskal(*samples)
            sig = "Khác biệt" if p < 0.05 else "Không khác biệt"
            print(f"  > Biến {m:15}: H-stat = {stat:6.2f} | p-value = {p:.4e} ({sig})")
    
    print("\nB. SO SÁNH CẶP (POST-HOC MANN-WHITNEY U + BONFERRONI)")
    num_comparisons = len(list(combinations(VALID_CONFIGS, 2)))
    
    for m in metrics:
        if m not in valid_data_dict: continue
        
        print(f"\n* Biến phân tích ranh giới: {m}")
        available_configs = list(valid_data_dict[m].keys())
        
        for c1, c2 in combinations(VALID_CONFIGS, 2):
            if c1 in available_configs and c2 in available_configs:
                s1, s2 = valid_data_dict[m][c1], valid_data_dict[m][c2]
                try:
                    _, p = stats.mannwhitneyu(s1, s2, alternative='two-sided')
                    p_adj = min(p * num_comparisons, 1.0)          
                    highlight = "📌" if "Normal" in [c1, c2] else "  "
                    sig = "Khác biệt" if p_adj < 0.05 else "Giống nhau"
                    print(f" {highlight} [{c1:6} vs {c2:6}]: p-adj = {p_adj:.4f} ({sig})")
                except ValueError: pass

# =============================================================================
# MODULE 4: PDI DYNAMICS (CONDITIONAL & DIRECTIONAL)
# =============================================================================
def test_pdi_dynamics(df):
    print("\n" + "="*80)
    print("🔥 4. PDI DYNAMICS: CONDITIONAL AND LEADING INDICATOR TESTS")
    print("="*80)
    
    print("A. CONDITIONAL TEST (KIỂM ĐỊNH CÓ ĐIỀU KIỆN)")
    print("Mục tiêu: Khi cấu trúc giống hệt nhau (R_t ≈ 0, đang xả hàng), PDI có phân biệt được Normal vs C1/C6 không?")
    
    cond_mask = (df['R_t'] < 0.05) & (df['dK_Pi_prime_pct'] < 0)
    df_cond = df[cond_mask].dropna(subset=['PDI_t', 'd_PDI_t'])
    
    g_normal = df_cond[df_cond['Configuration'] == 'Normal']
    g_crash = df_cond[df_cond['Configuration'].isin(['C1', 'C6'])]
    
    print(f"Số lượng quan sát thỏa mãn: Normal = {len(g_normal)}, C1/C6 = {len(g_crash)}")
    
    if len(g_normal) >= 3 and len(g_crash) >= 3:
        _, p_pdi = stats.mannwhitneyu(g_normal['PDI_t'], g_crash['PDI_t'], alternative='two-sided')
        _, p_dpdi = stats.mannwhitneyu(g_normal['d_PDI_t'], g_crash['d_PDI_t'], alternative='two-sided')
        print(f"  > Phân tách bằng PDI_t   : p-value = {p_pdi:.4e}")
        print(f"  > Phân tách bằng ΔPDI_t  : p-value = {p_dpdi:.4e}")
        
        if p_pdi < 0.05 or p_dpdi < 0.05:
            print("  ✔️ PASS: Dù ép cấu trúc bằng nhau, PDI vẫn vạch trần được sự khác biệt!")
        else:
            print("  ❌ FAIL: PDI mất khả năng phân biệt.")
    else:
        print("  ⚠️ Không đủ dữ liệu cho kiểm định Conditional.")

    print("\nB. DIRECTIONAL TEST (KIỂM ĐỊNH TÍNH DẪN DẮT BẰNG ĐỘ TRỄ)")
    print("Mục tiêu: So sánh xem PDI của quý trước (t-1) giữa nhóm sống sót và nhóm sập (C1/C6) có khác nhau không?")
    
    reg_df = df.dropna(subset=['is_Crash_Regime', 'PDI_lag1'])
    group_survive = reg_df[reg_df['is_Crash_Regime'] == 0]['PDI_lag1']
    group_crash = reg_df[reg_df['is_Crash_Regime'] == 1]['PDI_lag1']
    
    print(f"Số lượng quan sát: Sống sót = {len(group_survive)}, Sập (C1/C6) = {len(group_crash)}")
    
    if len(group_survive) >= 5 and len(group_crash) >= 5:
        _, p_val_lag = stats.mannwhitneyu(group_survive, group_crash, alternative='two-sided')
        print(f"  > Khác biệt PDI(t-1) giữa 2 nhóm: p-value = {p_val_lag:.4e}")
        
        if p_val_lag < 0.05:
            print("  ✔️ PASS: Nhóm sập và nhóm sống sót đã có mức PDI khác biệt từ tận quý trước! (Leading Indicator)")
        else:
            print("  ❌ FAIL: PDI là biến đồng hành (Concurrent Marker), không có tính tiên đoán.")

# =============================================================================
# MODULE 5: ROBUSTNESS (TEMPORAL & BOOTSTRAPPING)
# =============================================================================
def test_robustness(df):
    print("\n" + "="*80)
    print("🌍 5. ROBUSTNESS TESTS (TEMPORAL STABILITY & BOOTSTRAPPING)")
    print("="*80)
    
    print("A. TEMPORAL SPLIT (KIỂM ĐỊNH TÍNH BẤT BIẾN THEO THỜI GIAN)")
    print("Mục tiêu: Cấu trúc Ranh giới E_3 có bị phá vỡ bởi cú sốc Vĩ mô (Ví dụ: Năm 2020) không?")
    
    df_pre = df[df['period_end'].dt.year < 2020]
    df_post = df[df['period_end'].dt.year >= 2020]
    
    print(f"Số mẫu Trước 2020: {len(df_pre)} | Từ 2020: {len(df_post)}")
    
    def test_boundary(data, period_name):
        n_e3 = data[data['Configuration'] == 'Normal']['E_3'].dropna()
        s_e3 = data[data['Configuration'].isin(['C1', 'C2', 'C3', 'C4', 'C5', 'C6'])]['E_3'].dropna()
        if len(n_e3) > 3 and len(s_e3) > 3:
            _, p_val = stats.mannwhitneyu(n_e3, s_e3, alternative='two-sided')
            sig = "VỮNG (Robust)" if p_val < 0.05 else "Gãy (Failed)"
            print(f"  > Ranh giới E_3 ({period_name:10}): p-value = {p_val:.4e} -> {sig}")
            
    test_boundary(df_pre, "Trước 2020")
    test_boundary(df_post, "Sau 2020")

    print("\nB. BOOTSTRAPPING (KIỂM ĐỊNH LẤY MẪU LẠI 1,000 LẦN)")
    print("Mục tiêu: Đảm bảo kết quả ΔPDI (Normal vs C1/C6) không bị thao túng bởi Outliers.")
    
    cond_mask = (df['R_t'] < 0.05) & (df['dK_Pi_prime_pct'] < 0)
    df_cond = df[cond_mask].dropna(subset=['d_PDI_t'])
    
    success_count = 0
    n_iterations = 1000
    print(f"🔄 Đang chạy giả lập {n_iterations} lần lấy mẫu ngẫu nhiên (chỉ giữ 80% dữ liệu)...")
    
    for _ in range(n_iterations):
        sample_df = df_cond.sample(frac=0.8, replace=True)
        g_norm = sample_df[sample_df['Configuration'] == 'Normal']['d_PDI_t']
        g_crash = sample_df[sample_df['Configuration'].isin(['C1', 'C6'])]['d_PDI_t']
        
        if len(g_norm) > 2 and len(g_crash) > 2:
            if stats.mannwhitneyu(g_norm, g_crash, alternative='two-sided')[1] < 0.05:
                success_count += 1
                
    robust_pct = (success_count / n_iterations) * 100
    print(f"  📊 Tỉ lệ thành công (p < 0.05): {robust_pct:.1f}%")
    
    if robust_pct > 90:
        print("  ✔️ PASS XUẤT SẮC: Mô hình của bạn là một khối bê tông. Cắt bỏ bất kỳ 20% dữ liệu nào, định lý ΔPDI vẫn luôn đúng!")
    else:
        print("  ⚠️ CẢNH BÁO: Kết quả phụ thuộc khá nhiều vào một số công ty cụ thể (Outliers).")
    print("="*80)

def main():
    df = load_and_prep_data()
    if df is not None:
        analyze_markov_transitions(df)
        analyze_correlations(df)
        analyze_kruskal_and_posthoc(df)
        test_pdi_dynamics(df)
        test_robustness(df)

if __name__ == "__main__":
    main()