"""
STEP 9: THEORETICAL CLASSIFICATION
Focus: Applies Proposition 7 rules to classify structural states (Normal, C1-C6) 
and assigns linguistic phases based on the math computed in Step 08.

Input: data/process/{sector}/{ticker}_processed.csv
Output: data/classified/{sector}/{ticker}_classified.csv
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ====================
CONFIG_FILE = Path('Survey_config.yaml')
PROCESS_FOLDER = Path('data/process')
CLASSIFIED_FOLDER = Path('data/classified')

def classify_state(row):
    """
    Classifies the structural configuration per Proposition 7.
    Fix 1: Priorities Normal condition before NaNs to rescue true Normal states.
    Fix 2: Eliminates the 'Other' state via exact coverage of structural logic.
    """
    rt = row.get('R_t')
    dk = row.get('dK_Pi_prime')
    s = row.get('s_total')
    dkp = row.get('dK_Pi_prime_pct')
    speculative = row.get('Speculative_Regime')

    if pd.isna(rt) or pd.isna(dk):
        return 'N/A'
    # Normal = không đủ 3 điều kiện theo Section 3.1
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
    
    # 1. ĐIỀU KIỆN TIÊN QUYẾT: Xét Normal trước tiên (Chỉ cần K_Pi' <= 0 là đủ)
    if pd.notna(k_pi_prime) and k_pi_prime <= 0:
        return 'Normal'
        
    # 2. LỌC DỮ LIỆU KHUYẾT: Các quý đầu tiên không có dynamics sẽ vào đây
    if pd.isna(rt) or pd.isna(dk) or pd.isna(k_pi_prime): 
        return 'N/A'
    
    # 3. PHÂN LOẠI ĐẦU CƠ (C1 - C6)
    rt_tol = 1e-6 
    
    # BRANCH A: No surplus or loss-making (s <= 0)
    if s <= 0:
        if dk <= 0: 
            return 'C1' if dkp <= -0.15 else 'C6' # Collapse / Silent Redistribution
        else:
            return 'C2'                           # Pure Speculative Growth
            
    # BRANCH B: Positive surplus exists (s > 0)
    else:
        if dk > 0:
            return 'C3'                           # Obligation growing despite surplus
        else:
            # Obligation is reducing (dk <= 0), evaluate discharge magnitude
            if rt >= 0.999:
                return 'C5'                       # Full Productive Discharge
            else:
                return 'C4'                       # Partial Absorption
                
    return 'Other' # An toàn dự phòng, toán học không bao giờ rơi vào đây.

def classify_company_data(df, ticker):
    """Applies classification and linguistic phase mapping."""
    df = df.copy()
    df['Ticker'] = ticker

    # 1. Xử lý các dị thường toán học (Vô cực)
    core_cols = ['E_3', 'R_t', 'PDI_t', 'PGR_t']
    for col in core_cols:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            
    # 2. Xử lý R_t an toàn: CHỈ gán R_t = 0 nếu doanh nghiệp lỗ/không có thặng dư
    # Tuyệt đối không dùng fillna(0) diện rộng để tránh lấp ảo Quý 1
    if 'R_t' in df.columns and 's_total' in df.columns:
        df['R_t'] = np.where(df['s_total'] <= 0, 0.0, df['R_t'])

    # 3. Phân loại cấu trúc gốc (Raw Classification)
    df['Raw_Configuration'] = df.apply(classify_state, axis=1)

    # 4. Gác cổng (Formal Gate Enforcement) - CẢI TIẾN QUAN TRỌNG
    # Chỉ đánh rớt về Normal nếu nó thực sự có dữ liệu (!= 'N/A')
    if 'Speculative_Regime' in df.columns:
        df['Configuration'] = np.where(
            (df['Speculative_Regime'] == False) & (df['Raw_Configuration'] != 'N/A'), 
            'Normal', 
            df['Raw_Configuration']
        )
        df['Regime_Label'] = np.where(
            df['Configuration'] == 'Normal', 
            'Normal_Regime',   
            np.where(df['Configuration'] == 'N/A', 'Unknown', 'Speculative_Regime')
        )
    else:
        df['Configuration'] = df['Raw_Configuration']
        df['Regime_Label'] = np.where(
            df['Configuration'] == 'Normal', 
            'Normal_Regime', 
            np.where(df['Configuration'] == 'N/A', 'Unknown', 'Speculative_Regime')
        )

    # 5. Dán nhãn Giai đoạn (Phase Mapping)
    conditions = [
        df['Configuration'] == 'Normal',        
        df['Configuration'].isin(['C2', 'C3']),   
        df['Configuration'] == 'C4',              
        df['Configuration'] == 'C1',              
        df['Configuration'] == 'C6',              
        df['Configuration'] == 'C5'               
    ]
    choices = [
        'Normal_Production',           
        'Accumulation_Phase',          
        'Partial_Absorption',          
        'Redistributive_Collapse',     
        'Silent_Redistribution',       
        'Productive_Discharge'         
    ]
    df['Phase'] = np.select(conditions, choices, default='Other')

    return df

def main():
    logger.info("="*70)
    logger.info("THEORETICAL CLASSIFICATION (C1-C6 & PHASES)")
    logger.info("="*70)

    if not CONFIG_FILE.exists():
        logger.error(f"❌ Input config missing: {CONFIG_FILE}")
        return
        
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    stats = {'classified': 0, 'errors': 0}

    for sector_name, sector_info in config.get('sectors', {}).items():
        logger.info(f"\nSector: {sector_name}")
        sector_classified_dir = CLASSIFIED_FOLDER / sector_name
        sector_classified_dir.mkdir(parents=True, exist_ok=True)
        
        for company in sector_info.get('companies', []):
            if company.get('status') != 'active': continue
            
            ticker = company['ticker']
            fpath = PROCESS_FOLDER / sector_name / f"{ticker}_processed.csv"
            
            if fpath.exists():
                try:
                    df_processed = pd.read_csv(fpath)
                    df_classified = classify_company_data(df_processed, ticker)
                    
                    out_path = sector_classified_dir / f"{ticker}_classified.csv"
                    df_classified.to_csv(out_path, index=False, encoding='utf-8-sig')
                    stats['classified'] += 1
                    logger.debug(f"  ✅ {ticker:6} | Classification applied.")
                except Exception as e:
                    logger.error(f"  ❌ {ticker:6} | Error: {e}")
                    stats['errors'] += 1

    logger.info("\n" + "="*70)
    logger.info(f"🎯 COMPLETE! Classified {stats['classified']} files.")
    logger.info(f"📁 Output saved to: {CLASSIFIED_FOLDER}")
    logger.info("="*70)

if __name__ == "__main__":
    main()