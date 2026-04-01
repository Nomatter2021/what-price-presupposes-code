"""
STEP 12: THEORETICAL CLASSIFICATION
Focus: Strictly applies Proposition 7 rules and Formal Gates to classify 
structural states explicitly into 7 categories: ['Normal', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'].
Input: data/process (Mathematical outputs from Step 11)
Output: data/classified (Labeled data, fully preserved, ready for sequence filtering)
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

# ==================== CONFIGURATION ====================
PROCESS_FOLDER = Path('data/process')
CLASSIFIED_FOLDER = Path('data/classified')

# Create output directory for this specific module
CLASSIFIED_FOLDER.mkdir(parents=True, exist_ok=True)

def assign_raw_configuration(row):
    """Translates mathematical parameters strictly into raw structural codes (C1-C6)."""
    rt = row.get('R_t')
    dk = row.get('dK_Pi_prime')
    s = row.get('s_total')
    dkp = row.get('dK_Pi_prime_pct')

    if pd.isna(rt) or pd.isna(dk): return 'N/A'

    tol = 1e-6
    # Core logical conditions based on Proposition 7
    if abs(rt) <= tol and dk <= -tol and abs(s) <= tol: 
        return 'C1' if dkp <= -0.15 else 'C6'
    if abs(rt) <= tol and dk > tol and abs(s) <= tol: 
        return 'C2'
    if rt > tol and dk > tol and s > tol: 
        return 'C3'
    if tol < rt < 0.999 and dk <= -tol and s > tol: 
        return 'C4'
    if rt >= 0.999 and dk <= -tol and s > tol: 
        return 'C5'

    return 'Other'

def classify_company_data(df, ticker):
    """Applies Formal Gates to finalize Configuration (Normal + C1-C6) and Phase."""
    df = df.copy()
    df['Ticker'] = ticker

    # 1. Handle mathematical anomalies
    core_cols = ['E_3', 'R_t', 'PDI_t', 'PGR_t']
    for col in core_cols:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            
    # Restore R_t and PDI_t to 0 if NaN is caused by zero surplus (loss-making state)
    if 'R_t' in df.columns: df['R_t'] = df['R_t'].fillna(0)
    if 'PDI_t' in df.columns: df['PDI_t'] = df['PDI_t'].fillna(0)

    # =========================================================================
    # 2. RAW CONFIGURATION (Purely Mathematical C1-C6)
    # =========================================================================
    df['Raw_Configuration'] = df.apply(assign_raw_configuration, axis=1)

    # =========================================================================
    # 3. FINAL CONFIGURATION (Applying Speculative Gate to assign 'Normal')
    # =========================================================================
    if 'Speculative_Regime' in df.columns:
        df['Configuration'] = np.where(
            df['Speculative_Regime'] == False, 
            'Normal',                   # Ghi đè 'Normal' nếu không phải đầu cơ
            df['Raw_Configuration']     # Giữ nguyên C1-C6 nếu đúng là đầu cơ
        )
        df['Regime'] = np.where(
            df['Configuration'] == 'Normal', 
            'Normal_Regime',   
            'Speculative_Regime'         
        )
    else:
        df['Configuration'] = df['Raw_Configuration']
        df['Regime'] = 'Unknown'

    # =========================================================================
    # 4. PHASE MAPPING (Explicitly mapped to the 7 configurations)
    # =========================================================================
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
    print("="*70)
    print("STEP 12: FINAL CLASSIFICATION (NORMAL & C1-C6)")
    print("="*70)

    if not PROCESS_FOLDER.exists():
        print(f"❌ Input folder missing: {PROCESS_FOLDER}")
        return

    processed_count = 0
    for file in os.listdir(PROCESS_FOLDER):
        if not file.endswith('.csv'): continue
            
        ticker = file.split('_')[0]
        filepath = PROCESS_FOLDER / file
        
        try:
            df = pd.read_csv(filepath)
            df_classified = classify_company_data(df, ticker)
            
            # Export the fully labeled dataset
            out_path = CLASSIFIED_FOLDER / file
            df_classified.to_csv(out_path, index=False, encoding='utf-8-sig')
            print(f"  ✅ {ticker:6} | Classification successful.")
            processed_count += 1
        except Exception as e:
            print(f"  ❌ {ticker:6} | Error during classification: {e}")

    print("="*70)
    print(f"🎯 COMPLETE! Classified {processed_count} files.")
    print(f"📁 Output saved to: {CLASSIFIED_FOLDER}")
    print("="*70)

if __name__ == "__main__":
    main()