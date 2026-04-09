"""
STEP 12: THEORETICAL CLASSIFICATION (SỬA THEO CHUẨN FILE 11)
Focus: Strictly applies Proposition 7 rules and Formal Gates to classify 
structural states (Normal, C1-C6) and their linguistic phases.
Input: data/process (Mathematical outputs from Step 08/11)
Output: data/classified (Labeled data, fully preserved, ready for sequence filtering)
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

# ==================== CONFIGURATION & PATHS ====================
PROCESS_FOLDER = Path('data/process')
CLASSIFIED_FOLDER = Path('data/classified')

CLASSIFIED_FOLDER.mkdir(parents=True, exist_ok=True)

# ==================== CLASSIFICATION LOGIC (THEO FILE 11) ====================

def classify_state(row: pd.Series) -> str:
    """
    Phân loại cấu hình theo Proposition 7, dùng gate keeper (C1&C2 hoặc Speculative_Regime).
    Logic hoàn toàn theo file 11.
    """
    # Lấy các giá trị cần thiết
    rt = row.get('R_t')
    dk = row.get('dK_Pi_prime')
    s = row.get('s_total', 0.0)
    dkp = row.get('dK_Pi_prime_pct', 0.0)
    k_pi_prime = row.get('K_Pi_prime')
    k_pi_prime_lag = row.get('K_Pi_prime_lag')
    rev = row.get('Revenue', 0.0)
    
    # Các gate từ step 08/11
    c1 = row.get('Gate_C1', False)
    c2 = row.get('Gate_C2', False)
    speculative_step8 = row.get('Speculative_Regime', False)
    
    # 1. Không đủ dữ liệu động lực (kỳ đầu tiên)
    if pd.isna(k_pi_prime_lag):
        return 'N/A'
    
    # 2. Điều kiện đủ để rơi vào Normal (K_Pi' <= 0)
    if pd.notna(k_pi_prime) and k_pi_prime <= 0:
        return 'Normal'
    
    # 3. GATE KEEPER (theo file 11)
    # Nếu không thỏa mãn (C1&C2) hoặc Speculative_Regime thì là Normal
    if not ((c1 and c2) or speculative_step8):
        return 'Normal'
    
    # 4. Phân nhánh dựa trên thặng dư
    # BRANCH A: Không có thặng dư (s <= 0)
    if s <= 0:
        if dk <= 0:
            # Collapse nếu tốc độ giảm mạnh (dưới -15%), nếu không là Silent Redistribution
            return 'C1' if dkp <= -0.15 else 'C6'
        else:
            return 'C2'  # Pure Speculative Growth
    
    # BRANCH B: Có thặng dư (s > 0)
    else:
        if dk > 0:
            return 'C3'  # Obligation growing despite surplus
        else:
            # dk <= 0: nghĩa vụ đang giảm
            # CHỐT CHẶN C5: Chỉ khi thực sự có sản xuất (Revenue > 0) và R_t >= 0.999
            if rt >= 0.999 and rev > 0:
                return 'C5'  # Full Productive Discharge
            else:
                return 'C4'  # Partial Absorption
    
    return 'Other'  # fallback lý thuyết không xảy ra


def classify_company_data(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Áp dụng phân loại, gán nhãn chế độ và pha ngôn ngữ, loại bỏ dòng N/A."""
    df = df.copy()
    df['Ticker'] = ticker
    
    # 1. Xử lý các giá trị vô hạn
    core_cols = ['E_3', 'R_t', 'PDI_t', 'PGR_t']
    for col in core_cols:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    
    # 2. Đảm bảo R_t = 0 khi không có thặng dư
    if 'R_t' in df.columns and 's_total' in df.columns:
        df['R_t'] = np.where(df['s_total'] <= 0, 0.0, df['R_t'])
    
    # 3. Phân loại cấu hình thô
    df['Configuration'] = df.apply(classify_state, axis=1)
    
    # 4. Gán nhãn chế độ (Regime_Label) – phân biệt Normal_Regime và Rentier_Regime
    #    Dựa vào Revenue: >0 là sản xuất (M-C-M'), <=0 là thuần tuý M-M' (Rentier)
    df['Regime_Label'] = np.where(
        df['Configuration'] == 'Normal',
        np.where(df.get('Revenue', 0) <= 0, 'Rentier_Regime', 'Normal_Regime'),
        np.where(df['Configuration'] == 'N/A', 'Unknown', 'Speculative_Regime')
    )
    
    # 5. Gán pha ngôn ngữ (Phase) theo mapping của file 11
    conditions = [
        (df['Configuration'] == 'Normal') & (df.get('Revenue', 0) > 0),   # Normal + có sản xuất
        (df['Configuration'] == 'Normal') & (df.get('Revenue', 0) <= 0),  # Normal + không sản xuất (Rentier)
        df['Configuration'].isin(['C2', 'C3']),                           # Tích luỹ
        df['Configuration'] == 'C4',                                      # Hấp thụ một phần
        df['Configuration'] == 'C1',                                      # Sụp đổ tái phân phối
        df['Configuration'] == 'C6',                                      # Tái phân phối thầm lặng
        df['Configuration'] == 'C5'                                       # Giải phóng sản xuất
    ]
    choices = [
        'Normal_Production',        # Sản xuất bình thường
        'Steady_Rent_Seeking',      # Tìm kiếm địa tô ổn định
        'Accumulation_Phase',       # Pha tích luỹ
        'Partial_Absorption',       # Hấp thụ một phần
        'Redistributive_Collapse',  # Sụp đổ tái phân phối
        'Silent_Redistribution',    # Tái phân phối thầm lặng
        'Productive_Discharge'      # Giải phóng sản xuất
    ]
    df['Phase'] = np.select(conditions, choices, default='Other')
    
    # 6. Loại bỏ các dòng không phân loại được (N/A) – giống file 11
    df_clean = df[df['Configuration'] != 'N/A'].copy().reset_index(drop=True)
    
    return df_clean


# ==================== MAIN EXECUTION ====================

def main() -> None:
    print("="*70)
    print("STEP 12: THEORETICAL CLASSIFICATION (ĐÃ SỬA THEO CHUẨN FILE 11)")
    print("="*70)
    
    if not PROCESS_FOLDER.exists():
        print(f"❌ Input folder missing: {PROCESS_FOLDER}")
        return
    
    processed_count = 0
    for file in os.listdir(PROCESS_FOLDER):
        if not file.endswith('.csv'):
            continue
        
        ticker = file.split('_')[0]  # Lấy ticker từ tên file (vd: AAPL_raw.csv -> AAPL)
        filepath = PROCESS_FOLDER / file
        
        try:
            df = pd.read_csv(filepath)
            df_classified = classify_company_data(df, ticker)
            
            out_path = CLASSIFIED_FOLDER / file
            df_classified.to_csv(out_path, index=False, encoding='utf-8-sig')
            print(f"  ✅ {ticker:6} | Classification successful (kept {len(df_classified)} rows, dropped N/A).")
            processed_count += 1
        except Exception as e:
            print(f"  ❌ {ticker:6} | Error during classification: {e}")
    
    print("="*70)
    print(f"🎯 COMPLETE! Classified {processed_count} files.")
    print(f"📁 Output saved to: {CLASSIFIED_FOLDER}")
    print("="*70)


if __name__ == "__main__":
    main()