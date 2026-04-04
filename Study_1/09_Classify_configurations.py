"""
STEP 12: THEORETICAL CLASSIFICATION
Focus: Strictly applies Proposition 7 rules and Formal Gates to classify 
structural states (Normal, C1-C6) and their linguistic phases.
Input: data/process (Mathematical outputs from Step 11)
Output: data/classified (Labeled data, fully preserved, ready for sequence filtering)
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

# ==================== CONFIGURATION & PATHS ====================
PROCESS_FOLDER = Path('data/process')
CLASSIFIED_FOLDER = Path('data/classified')

# Ensure output directory exists
CLASSIFIED_FOLDER.mkdir(parents=True, exist_ok=True)

# ==================== CLASSIFICATION LOGIC ====================

def assign_raw_configuration(row: pd.Series) -> str:
    """
    Classifies the structural configuration per Proposition 7.
    Logic space is completely mapped. 'Other' cases are mathematically eliminated.
    """
    k_pi_prime = row.get('K_Pi_prime')
    rt = row.get('R_t')
    dk = row.get('dK_Pi_prime')
    s = row.get('s_total')
    dkp = row.get('dK_Pi_prime_pct')

    # 1. PREREQUISITE: Evaluate Normal state first (K_Pi' <= 0 is sufficient)
    if pd.notna(k_pi_prime) and k_pi_prime <= 0:
        return 'Normal'

    # 2. MISSING DATA FILTER: Initial quarters lacking dynamics fall here
    if pd.isna(rt) or pd.isna(dk) or pd.isna(k_pi_prime): 
        return 'N/A'

    # 3. SPECULATIVE CLASSIFICATION (C1 - C6)
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

    return 'Other' # Mathematical fallback


def classify_company_data(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Applies Formal Gates and linguistic phase mapping."""
    df = df.copy()
    df['Ticker'] = ticker

    # 1. Handle mathematical anomalies (Infinity bounds)
    core_cols = ['E_3', 'R_t', 'PDI_t', 'PGR_t']
    for col in core_cols:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            
    # 2. Safe R_t handling: Force R_t = 0 if loss-making / no surplus
    if 'R_t' in df.columns and 's_total' in df.columns:
        df['R_t'] = np.where(df['s_total'] <= 0, 0.0, df['R_t'])

    # 3. Raw Classification Extraction
    df['Raw_Configuration'] = df.apply(assign_raw_configuration, axis=1)

    # 4. Formal Gate Enforcement (Synced with Step 11 outputs)
    if 'Speculative_Regime' in df.columns:
        df['Configuration'] = np.where(
            (df['Speculative_Regime'] == False) & (df['Raw_Configuration'] != 'N/A'), 
            'Normal', 
            df['Raw_Configuration']
        )
    else:
        df['Configuration'] = df['Raw_Configuration']
        
    df['Regime_Label'] = np.where(
        df['Configuration'] == 'Normal', 
        'Normal_Regime',   
        np.where(df['Configuration'] == 'N/A', 'Unknown', 'Speculative_Regime')
    )

    # 5. Linguistic Phase Mapping
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


# ==================== MAIN EXECUTION ====================

def main() -> None:
    print("="*70)
    print("STEP 12: THEORETICAL CLASSIFICATION (C1-C6 & PHASES)")
    print("="*70)

    if not PROCESS_FOLDER.exists():
        print(f"❌ Input folder missing: {PROCESS_FOLDER}")
        return

    processed_count = 0
    for file in os.listdir(PROCESS_FOLDER):
        if not file.endswith('.csv'): 
            continue
            
        ticker = file.split('_')[0]
        filepath = PROCESS_FOLDER / file
        
        try:
            df = pd.read_csv(filepath)
            df_classified = classify_company_data(df, ticker)
            
            # Export the fully labeled dataset WITHOUT dropping any rows
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
