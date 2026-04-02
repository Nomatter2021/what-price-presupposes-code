"""
STEP 08: SPECULATIVE FRAMEWORK MATHEMATICAL CALCULATION
Focus: Strictly computes LTV metrics (V_Prod_base, E*, K_Pi', R_t, PDI_t) and Formal Gates.
Note: Classification (C1-C6 and Normal) is strictly delegated to Step 09.
"""

import pandas as pd
import numpy as np
import os
import yaml
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
CONFIG_FILE = Path('survey_config.yaml')
RAW_DATA_FOLDER = Path('data/raw')
PROCESS_FOLDER = Path('data/process')
BENCHMARK_FOLDER = Path('data/processed')

PROCESS_FOLDER.mkdir(parents=True, exist_ok=True)

def get_q_period(date):
    """Convert datetime to Quarter period string (e.g., '2023Q1')."""
    if pd.isna(date): return None
    return (pd.to_datetime(date) - pd.Timedelta(days=15)).to_period('Q')

def load_benchmark_lookup():
    """Load the smoothed 3-year rolling benchmark margins from Step 03."""
    lookup = {}
    sectors = ['Technology', 'Retail', 'Services']
    
    for sector in sectors:
        csv_path = BENCHMARK_FOLDER / f"{sector}_benchmark_median.csv"
        xlsx_path = BENCHMARK_FOLDER / f"{sector}_benchmark_median.xlsx"
        
        df = None
        if csv_path.exists():
            df = pd.read_csv(csv_path)
        elif xlsx_path.exists():
            df = pd.read_excel(xlsx_path)
            
        if df is not None:
            if 'usable_for_benchmark' in df.columns:
                df = df[df['usable_for_benchmark'] == True].copy()
            if 'period_end' in df.columns and 'Operating_Margin_median' in df.columns:
                df['period_end'] = pd.to_datetime(df['period_end'])
                for _, row in df.iterrows():
                    if pd.notna(row['Operating_Margin_median']):
                        q_period = get_q_period(row['period_end'])
                        lookup[(sector, q_period)] = row['Operating_Margin_median']
                        
    return lookup

def get_benchmark_margin(sector, period_end, lookup):
    """Retrieve the corresponding benchmark margin, falling back to the closest past value."""
    target_q = get_q_period(period_end)
    if (sector, target_q) in lookup:
        return lookup[(sector, target_q)]
    
    candidates = [v for k, v in lookup.items() if k[0] == sector and k[1] <= target_q]
    return candidates[-1] if candidates else np.nan

def calculate_framework_metrics(df, sector, benchmark_lookup):
    """Calculate core LTV and Speculative Regime metrics strictly (No Classification)."""
    df = df.copy()
    df['period_end'] = pd.to_datetime(df['period_end'])
    df = df.sort_values('period_end').reset_index(drop=True)

    rev_col = next((c for c in df.columns if str(c).lower() == 'revenue'), None)
    if rev_col and rev_col != 'Revenue':
        df.rename(columns={rev_col: 'Revenue'}, inplace=True)

    required_cols = ['Revenue', 'market_cap', 'KBrand']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"  ⚠️ Thiếu cột bắt buộc: {missing_cols}")
        return None

    # 1. Logic Operating Margin & Opex
    rnd = next((c for c in df.columns if 'research' in c.lower()), None)
    sga = next((c for c in df.columns if 'selling' in c.lower()), None)
    df['Opex_Ratio'] = (df[rnd].fillna(0) + df[sga].fillna(0)) / df['Revenue'] if rnd and sga else 0
    
    if 'Operating_Margin' not in df.columns:
        df['Operating_Margin'] = (df['Revenue'] - df.get('CostOfRevenue', 0)) / df['Revenue'] - 0.2

    use_gross = (df['Operating_Margin'] < 0) & (df['Opex_Ratio'] > 0.30) & (df.get('Gross_Margin', -1) > 0)
    df['Selected_Margin'] = np.where(use_gross, df.get('Gross_Margin', df['Operating_Margin']), df['Operating_Margin'])

    # 2. Tính toán Benchmark và Thặng dư (Surplus)
    df['Benchmark_Margin'] = [get_benchmark_margin(sector, d, benchmark_lookup) for d in df['period_end']]
    
    df['s_baseline_value'] = np.where(df['Benchmark_Margin'].isna(), 
                                      df['Revenue'] * df['Selected_Margin'].clip(lower=0),
                                      df['Revenue'] * np.minimum(df['Selected_Margin'].clip(lower=0), df['Benchmark_Margin']))
    
    df['S_Surplus'] = np.where(df['Benchmark_Margin'].isna(), 0.0,
                               df['Revenue'] * np.maximum(0, df['Selected_Margin'] - df['Benchmark_Margin']))
    
    df['s_total'] = df['s_baseline_value'] + df['S_Surplus']

    # 3. Bóc tách K_Pi' 
    df['V_Prod_base'] = df['Revenue'] * (1 - df['Selected_Margin'].clip(lower=0))
    df['K_Pi_prime'] = df['market_cap'] - (df['V_Prod_base'] + df['s_total'] + df['KBrand'])
    
    # 4. Các tỷ số phi thứ nguyên (Dimensionless Ratios: E*)
    vpb = df['V_Prod_base']
    df['E_star'] = np.where(vpb > 0, (df['market_cap'] - vpb) / vpb, np.nan)
    df['E_0'] = np.where(vpb > 0, df['s_baseline_value'] / vpb, np.nan)
    df['E_1'] = np.where(vpb > 0, df['S_Surplus'] / vpb, np.nan)
    df['E_2'] = np.where(vpb > 0, df['KBrand'] / vpb, np.nan)
    df['E_3'] = np.where(vpb > 0, df['K_Pi_prime'] / vpb, np.nan)
    
    # 5. CÁC CHỈ SỐ ĐỘNG HỌC VÀ ĐỘNG LƯỢNG (DYNAMICS) - ĐÃ FILL 0 ĐỂ FIX GATE_C3
    df['K_Pi_prime_lag'] = df['K_Pi_prime'].shift(1)
    
    # Fill 0 cho R_t nếu là quý đầu tiên (lag = NaN)
    df['R_t'] = np.where(
        df['K_Pi_prime_lag'].notna() & (df['K_Pi_prime_lag'] != 0), 
        df['s_total'] / df['K_Pi_prime_lag'], 
        0.0  # <--- Bắt buộc bằng 0
    )
    
    df['T_t'] = np.where(df['R_t'] > 1e-6, 1 / df['R_t'], np.inf)
    
    # Fill 0 cho dK (vì quý đầu không có thay đổi)
    df['dK_Pi_prime'] = df['K_Pi_prime'].diff().fillna(0.0) # <--- Bắt buộc fillna(0)
    
    # Fill 0 cho dK_pct
    df['dK_Pi_prime_pct'] = np.where(
        df['K_Pi_prime_lag'].notna() & (df['K_Pi_prime_lag'].abs() > 0), 
        df['dK_Pi_prime'] / df['K_Pi_prime_lag'].abs(), 
        0.0  # <--- Bắt buộc bằng 0
    )
    
    denominator = df['dK_Pi_prime'].abs() + df['s_total']
    df['PDI_t'] = np.where(
        (denominator != 0) & denominator.notna(), 
        df['s_total'] / denominator, 
        0.0
    )
    
    df['PGR_t'] = df['V_Prod_base'].pct_change().fillna(0.0)

    # 6. Formal Gates: Xác định chế độ Vĩ mô (Normal vs Speculative)
    df['Gate_C1'] = (df['K_Pi_prime'] / df['market_cap']) > 0.5
    df['Gate_C2'] = df['E_3'] > (df['E_0'] + df['E_1'] + df['E_2'])
    
    # R_t đã được fill 0 ở quý đầu, nên Gate_C3 sẽ pass (True) hợp lệ
    df['Gate_C3'] = df['R_t'] < 1.0 
    
    df['Speculative_Regime'] = df['Gate_C1'] & df['Gate_C2'] & df['Gate_C3']

    return df

def main():
    print("="*70)
    print("STEP 08: FRAMEWORK MATHEMATICAL KERNEL (E*, K_Pi', R_t, PDI_t)")
    print("="*70)

    if not CONFIG_FILE.exists():
        print(f"❌ Lỗi: Không tìm thấy file {CONFIG_FILE}")
        return

    benchmark_lookup = load_benchmark_lookup()
    
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    processed_count = 0
    
    for sector_name, sector_info in config.get('sectors', {}).items():
        print(f"\n📑 Sector: {sector_name}")
        
        for company in sector_info.get('companies', []):
            ticker = company.get('ticker')
            
            if company.get('status') != 'active':
                continue

            comp_type = company.get('type', 'Focal').replace('Sa', '')
            fname = f"{ticker}_raw.csv"
            fpath = RAW_DATA_FOLDER / fname
            
            if fpath.exists():
                df_raw = pd.read_csv(fpath)
                df_out = calculate_framework_metrics(df_raw, sector_name, benchmark_lookup)
                
                if df_out is not None:
                    out_path = PROCESS_FOLDER / fname
                    df_out.to_csv(out_path, index=False, encoding='utf-8-sig')
                    print(f"  ✅ {ticker:6} | Toán học hoàn tất và đã xuất file.")
                    processed_count += 1
                else:
                    print(f"  ❌ {ticker:6} | Lỗi: Không thể tính toán (Thiếu cột cốt lõi).")
            else:
                print(f"  ❌ {ticker:6} | Không tìm thấy file gốc tại: {fpath}")

    print("\n" + "="*70)
    print(f"🎯 KẾT THÚC! Đã xử lý thành công toán học cho {processed_count} công ty.")
    print("="*70)

if __name__ == "__main__":
    main()