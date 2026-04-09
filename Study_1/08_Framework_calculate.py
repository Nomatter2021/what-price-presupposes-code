"""
STEP 08: SPECULATIVE FRAMEWORK MATHEMATICAL KERNEL
Focus: Strictly computes Labour Theory of Value (LTV) metrics (V_Prod_base, E*, K_Pi', R_t, PDI_t) 
and Formal Gates.
Note: Configuration classification (C1-C6 and Normal) is strictly delegated to Step 09.
Sửa theo chuẩn của file 10_Framework_calculate.py
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import warnings
from typing import Dict, Tuple, Optional

warnings.filterwarnings('ignore')

# ==================== CONFIGURATION & PATHS ====================
CONFIG_FILE = Path('survey_config.yaml')
RAW_DATA_FOLDER = Path('data/raw')
PROCESS_FOLDER = Path('data/process')
BENCHMARK_FOLDER = Path('data/processed')

PROCESS_FOLDER.mkdir(parents=True, exist_ok=True)

# ==================== HELPER FUNCTIONS ====================

def get_q_period(date: pd.Timestamp) -> Optional[pd.Period]:
    """Shift 15 ngày rồi lấy quý (theo chuẩn file 10)"""
    if pd.isna(date):
        return None
    return (pd.to_datetime(date) - pd.Timedelta(days=15)).to_period('Q')


def load_benchmark_lookup() -> Dict[Tuple[str, pd.Period], float]:
    """
    Load tất cả benchmark files từ thư mục (linh hoạt, không cứng 3 sector).
    Giống logic file 10.
    """
    lookup = {}
    if not BENCHMARK_FOLDER.exists():
        return lookup
    
    for file_path in BENCHMARK_FOLDER.glob("*_benchmark_median.xlsx"):
        sector = file_path.stem.replace("_benchmark_median", "")
        try:
            df = pd.read_excel(file_path)
            if 's_baseline' in df.columns and 'period_end' in df.columns:
                df['period_end'] = pd.to_datetime(df['period_end'])
                for _, row in df.iterrows():
                    if 'Operating_Margin_median' in df.columns:
                        q_period = get_q_period(row['period_end'])
                        lookup[(sector, q_period)] = row['Operating_Margin_median']
        except Exception as e:
            print(f"  ⚠️ Lỗi đọc benchmark {sector}: {e}")
    return lookup


def get_benchmark_margin(sector: str, period_end: pd.Timestamp, lookup: Dict) -> float:
    """Lấy benchmark margin, fallback về quý gần nhất trong quá khứ"""
    target_q = get_q_period(period_end)
    if (sector, target_q) in lookup:
        return lookup[(sector, target_q)]
    
    candidates = [v for k, v in lookup.items() if k[0] == sector and k[1] <= target_q]
    return candidates[-1] if candidates else np.nan


# ==================== CORE MATHEMATICAL ENGINE ====================

def calculate_framework_metrics(df: pd.DataFrame, sector: str, benchmark_lookup: Dict) -> Optional[pd.DataFrame]:
    """
    Tính toán các chỉ số LTV và Formal Gates theo đúng phương pháp luận của file 10.
    """
    df = df.copy()
    df['period_end'] = pd.to_datetime(df['period_end'])
    df = df.sort_values('period_end').reset_index(drop=True)

    # Chuẩn hóa tên cột Revenue
    rev_col = next((c for c in df.columns if str(c).lower() == 'revenue'), None)
    if rev_col and rev_col != 'Revenue':
        df.rename(columns={rev_col: 'Revenue'}, inplace=True)

    required_cols = ['Revenue', 'market_cap', 'KBrand']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"  ⚠️ Thiếu cột bắt buộc: {missing_cols}")
        return None

    # ========== 1. Phân biệt nhóm productive (M-C-M') vs non-productive (M-M') ==========
    df['Revenue'] = pd.to_numeric(df['Revenue'], errors='coerce').fillna(0)
    is_productive = df['Revenue'] > 0   # Dấu hiệu vòng M-C-M'

    # Tìm cột Operating Income (Lợi nhuận) cho nhóm M-M'
    op_inc_col = next((c for c in df.columns if 'operating' in str(c).lower() and 'income' in str(c).lower()), None)
    if not op_inc_col:
        op_inc_col = next((c for c in df.columns if 'operatingincome' in str(c).lower()), None)
    
    if op_inc_col:
        df[op_inc_col] = pd.to_numeric(df[op_inc_col], errors='coerce').fillna(0)
    else:
        df['Fallback_OpInc'] = 0.0
        op_inc_col = 'Fallback_OpInc'

    # ========== 2. Xác định Margin được chọn ==========
    rnd_col = next((c for c in df.columns if 'research' in c.lower()), None)
    sga_col = next((c for c in df.columns if 'selling' in c.lower()), None)
    
    df['Opex_Ratio'] = 0.0
    if rnd_col and sga_col:
        df['Opex_Ratio'] = (df[rnd_col].fillna(0) + df[sga_col].fillna(0)) / df['Revenue'].replace(0, np.nan)
        df['Opex_Ratio'] = df['Opex_Ratio'].fillna(0.0)

    if 'Operating_Margin' not in df.columns:
        # Fallback nếu không có Operating_Margin
        df['Operating_Margin'] = np.nan

    # Gross Margin nếu có
    if 'Gross_Margin' not in df.columns:
        df['Gross_Margin'] = np.nan

    # Ưu tiên Gross Margin nếu Operating Margin âm và Opex quá cao
    use_gross = (df['Operating_Margin'] < 0) & (df['Opex_Ratio'] > 0.30) & (df['Gross_Margin'] > 0)
    df['Selected_Margin'] = np.where(use_gross, df['Gross_Margin'], df['Operating_Margin'])
    
    # Cap an toàn (theo file 10)
    safe_margin = df['Selected_Margin'].clip(lower=None, upper=0.9999)

    # ========== 3. Benchmark và các thành phần surplus ==========
    df['Benchmark_Margin'] = [get_benchmark_margin(sector, d, benchmark_lookup) for d in df['period_end']]
    
    # V_Prod_base: chỉ có ý nghĩa với nhóm productive, nếu không thì = 0
    df['V_Prod_base'] = np.where(is_productive, df['Revenue'] * (1 - safe_margin.clip(lower=0)), 0.0)
    
    # s_baseline_value
    s_base_calc = np.where(
        df['Benchmark_Margin'].isna(),
        df['Revenue'] * safe_margin.clip(lower=0),
        df['Revenue'] * np.minimum(safe_margin.clip(lower=0), df['Benchmark_Margin'])
    )
    df['s_baseline_value'] = np.where(is_productive, s_base_calc, 0.0)
    
    # S_Surplus: 
    # - Với nhóm productive: tính từ benchmark margin
    # - Với nhóm M-M': lấy trực tiếp Operating Income (lợi nhuận) làm surplus
    s_surp_calc = np.where(
        df['Benchmark_Margin'].isna(),
        0.0,
        df['Revenue'] * np.maximum(0, safe_margin - df['Benchmark_Margin'])
    )
    df['S_Surplus'] = np.where(is_productive, s_surp_calc, df[op_inc_col].clip(lower=0))
    
    df['s_total'] = df['s_baseline_value'] + df['S_Surplus']

    # ========== 4. K_Pi' (Phần dư chưa được khởi tạo) ==========
    df['K_Pi_prime'] = df['market_cap'] - (df['V_Prod_base'] + df['s_total'] + df['KBrand'])

    # ========== 5. Các tỷ lệ E* (chỉ tính khi V_Prod_base > 0) ==========
    vpb = df['V_Prod_base']
    valid_base = vpb > 0
    
    df['E_star'] = np.where(valid_base, (df['market_cap'] - vpb) / vpb, np.nan)
    df['E_0'] = np.where(valid_base, df['s_baseline_value'] / vpb, np.nan)
    df['E_1'] = np.where(valid_base, df['S_Surplus'] / vpb, np.nan)
    df['E_2'] = np.where(valid_base, df['KBrand'] / vpb, np.nan)
    df['E_3'] = np.where(valid_base, df['K_Pi_prime'] / vpb, np.nan)

    # ========== 6. Động lực và Momentum ==========
    df['K_Pi_prime_lag'] = df['K_Pi_prime'].shift(1)
    
    # R_t (tỷ lệ hấp thụ)
    df['R_t'] = np.where(
        df['K_Pi_prime_lag'].notna() & (df['K_Pi_prime_lag'] != 0),
        df['s_total'] / df['K_Pi_prime_lag'],
        0.0
    )
    
    # dK và PDI_t
    df['dK_Pi_prime'] = df['K_Pi_prime'].diff().fillna(0.0)
    df['dK_Pi_prime_pct'] = np.where(
        df['K_Pi_prime_lag'].notna() & (df['K_Pi_prime_lag'].abs() > 0),
        df['dK_Pi_prime'] / df['K_Pi_prime_lag'].abs(),
        0.0
    )
    
    df['PGR_t'] = np.where(is_productive, df['V_Prod_base'].pct_change().fillna(0.0), 0.0)
    
    denominator = df['dK_Pi_prime'].abs() + df['s_total']
    df['PDI_t'] = np.where(
        (denominator != 0) & denominator.notna(),
        df['s_total'] / denominator,
        0.0
    )

    # ========== 7. Formal Gates (theo đúng file 10) ==========
    # Gate_C1: chỉ xét khi V_Prod_base > 0, so sánh K_Pi' / V_Prod_base > 0
    df['Gate_C1'] = np.where(
        df['V_Prod_base'] > 0,
        (df['K_Pi_prime'] / df['V_Prod_base']) > 0,
        False
    )
    
    # Gate_C2: E_3 > E_0 + E_1 + E_2
    df['Gate_C2'] = (df['E_3'] > (df['E_0'] + df['E_1'] + df['E_2'])).astype(bool)
    
    # Gate_C3: R_t < 1.0
    df['Gate_C3'] = (df['R_t'] < 1.0).astype(bool)
    
    standard_speculative = df['Gate_C1'] & df['Gate_C2'] & df['Gate_C3']
    
    # Nhận diện tư bản giả (extreme speculation): không có sản xuất nhưng có vốn hóa
    extreme_speculation = (df['Revenue'] <= 0) & (df['market_cap'] > 0)
    df['Speculative_Regime'] = np.where(extreme_speculation, True, standard_speculative)

    return df


# ==================== MAIN EXECUTION ====================

def main() -> None:
    print("="*70)
    print("STEP 08: FRAMEWORK MATHEMATICAL KERNEL (ĐÃ SỬA THEO CHUẨN FILE 10)")
    print("="*70)

    if not CONFIG_FILE.exists():
        print(f"❌ Lỗi: Không tìm thấy file cấu hình {CONFIG_FILE}")
        return

    benchmark_lookup = load_benchmark_lookup()
    
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    processed_count = 0
    
    for sector_name, sector_info in config.get('sectors', {}).items():
        print(f"\n📑 Xử lý ngành: {sector_name}")
        
        for company in sector_info.get('companies', []):
            ticker = company.get('ticker')
            
            if company.get('status') != 'active':
                continue

            fname = f"{ticker}_raw.csv"
            fpath = RAW_DATA_FOLDER / fname
            
            if fpath.exists():
                try:
                    df_raw = pd.read_csv(fpath)
                    df_out = calculate_framework_metrics(df_raw, sector_name, benchmark_lookup)
                    
                    if df_out is not None:
                        out_path = PROCESS_FOLDER / fname
                        df_out.to_csv(out_path, index=False, encoding='utf-8-sig')
                        print(f"  ✅ {ticker:6} | Tính toán thành công.")
                        processed_count += 1
                    else:
                        print(f"  ❌ {ticker:6} | Lỗi: Thiếu cột bắt buộc.")
                except Exception as e:
                    print(f"  ❌ {ticker:6} | Lỗi ngoại lệ: {str(e)}")
            else:
                print(f"  ❌ {ticker:6} | Không tìm thấy file: {fpath}")

    print("\n" + "="*70)
    print(f"🎯 HOÀN TẤT! Đã xử lý thành công {processed_count} thực thể.")
    print("="*70)


if __name__ == "__main__":
    main()