"""
STEP 09: THEORETICAL CLASSIFICATION
Objective: Applies Proposition 7 to classify structural configurations on a strictly discrete quarterly basis.
Methodology: Evaluates quarterly financial dynamics to map observations into Normal or Speculative states (C1-C6).
Input: data/process/*.csv
Output: data/classified/*.csv
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ====================
PROCESS_FOLDER = Path('data/process')
CLASSIFIED_FOLDER = Path('data/classified')

# Ensure output directory exists
CLASSIFIED_FOLDER.mkdir(parents=True, exist_ok=True)

def classify_state(row: pd.Series) -> str:
    """
    Evaluates the structural configuration for a single quarterly observation per Proposition 7.
    """
    rt = row.get('R_t')
    dk = row.get('dK_Pi_prime')
    s = row.get('s_total')
    dkp = row.get('dK_Pi_prime_pct')
    k_pi_prime = row.get('K_Pi_prime')
    speculative = row.get('Speculative_Regime')

    # Prerequisite evaluation: Prioritize Normal condition to rescue true Normal states
    if not speculative or (pd.notna(k_pi_prime) and k_pi_prime <= 0):
        return 'Normal'

    # Missing data filtration: Identifies initial quarters lacking dynamic transition data
    if pd.isna(rt) or pd.isna(dk):
        return 'N/A'

    rt_tol = 1e-6

    # Speculative Classification Logic (C1 - C6) evaluated quarter-by-quarter
    if s <= 0:  # Branch A: No surplus or loss-making
        if dk <= 0:
            return 'C1' if dkp <= -0.15 else 'C6' # Redistributive Collapse or Silent Redistribution
        else:
            return 'C2'                           # Pure Speculative Growth (Gestation)
    else:       # Branch B: Positive surplus exists
        if dk > 0:
            return 'C3'                           # Obligation growing despite surplus
        else:
            if rt >= 0.999:
                return 'C5'                       # Full Productive Discharge
            if rt_tol < rt < 0.999:
                return 'C4'                       # Partial Absorption

    return 'Other' # Mathematical fallback (should theoretically be unreachable)

def classify_company_data(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Applies classification logic across longitudinal quarterly data and assigns linguistic phases."""
    df = df.copy()
    df['Ticker'] = ticker

    # Isolate mathematical anomalies (e.g., division by zero resulting in infinity)
    core_cols = ['E_3', 'R_t', 'PDI_t', 'PGR_t']
    for col in core_cols:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            
    # Safe R_t handling: Force R_t = 0 only if strictly loss-making to prevent false early-quarter interpolation
    if 'R_t' in df.columns and 's_total' in df.columns:
        df['R_t'] = np.where(df['s_total'] <= 0, 0.0, df['R_t'])

    # Execute quarter-level classification
    df['Raw_Configuration'] = df.apply(classify_state, axis=1)

    # Formal Gate Enforcement: Override missing speculative flags if mathematically Normal
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

    # Linguistic Phase Mapping corresponding to structural configurations
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
    """Batch executes the quarterly classification logic directly across the process folder."""
    logger.info("="*70)
    logger.info("QUARTERLY THEORETICAL CLASSIFICATION (C1-C6 & PHASES)")
    logger.info("="*70)

    if not PROCESS_FOLDER.exists():
        logger.error(f"❌ Input folder missing: {PROCESS_FOLDER}")
        return

    stats = {'classified': 0, 'errors': 0}

    for file in os.listdir(PROCESS_FOLDER):
        if not file.endswith('.csv'): 
            continue
            
        ticker = file.split('_')[0]
        filepath = PROCESS_FOLDER / file
        
        try:
            df = pd.read_csv(filepath)
            df_classified = classify_company_data(df, ticker)
            
            # Export the fully labeled dataset
            out_path = CLASSIFIED_FOLDER / file
            df_classified.to_csv(out_path, index=False, encoding='utf-8-sig')
            
            logger.debug(f"  ✅ {ticker:6} | Classification successful.")
            stats['classified'] += 1
            
        except Exception as e:
            logger.error(f"  ❌ {ticker:6} | Error during classification: {e}")
            stats['errors'] += 1

    logger.info("\n" + "="*70)
    logger.info(f"🎯 COMPLETE! Classified {stats['classified']} longitudinal files.")
    logger.info(f"📁 Output saved to: {CLASSIFIED_FOLDER}")
    logger.info("="*70)

if __name__ == "__main__":
    main()