"""
STEP 06: DATA CLEANING & CONTINUITY FILTER
Logic: Scans the entire time series to identify the longest continuous sequence.
If the maximum contiguous sequence is < 6 quarters, the entity is structurally disqualified.
CRITICAL: Updates 'survey_config.yaml' status machine to 'delete' for downstream exclusion.
"""

import pandas as pd
import os
import yaml
import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any

# ==================== CONFIGURATION ====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

CONFIG_FILE = Path('survey_config.yaml')
DATA_FOLDER = Path('data/raw')

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Removes duplicate columns and standardizes core variable nomenclature."""
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Resolve multiple price columns
    price_cols = [c for c in df.columns if 'price_at_period' in c.lower()]
    if len(price_cols) >= 1:
        best_price = max(price_cols, key=lambda c: df[c].notna().sum())
        df = df.drop(columns=[c for c in price_cols if c != best_price])
        df.rename(columns={best_price: 'price_at_period_end'}, inplace=True)
    
    # Resolve multiple shares columns
    shares_cols = [c for c in df.columns if 'shares_outstanding' in c.lower()]
    if len(shares_cols) >= 1:
        best_shares = max(shares_cols, key=lambda c: df[c].notna().sum())
        df = df.drop(columns=[c for c in shares_cols if c != best_shares])
        df.rename(columns={best_shares: 'shares_outstanding'}, inplace=True)
    
    # Recalculate missing market caps if components exist
    if 'price_at_period_end' in df.columns and 'shares_outstanding' in df.columns:
        if 'market_cap' not in df.columns:
            df['market_cap'] = None
        mask = df['market_cap'].isna() & df['shares_outstanding'].notna() & df['price_at_period_end'].notna()
        df.loc[mask, 'market_cap'] = (df.loc[mask, 'price_at_period_end'] * df.loc[mask, 'shares_outstanding']).round(2)
    
    return df

def count_consecutive_quarters(df: pd.DataFrame, min_consecutive: int = 6) -> Tuple[int, Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """Identifies the longest contiguous block of quarters with valid fundamental data."""
    if 'period_end' not in df.columns:
        return 0, None, None
    
    df = df.copy()
    df['period_end'] = pd.to_datetime(df['period_end'])
    
    valid_rev_names = ['revenue', 'totalrevenue', 'revenues', 'netsales']
    revenue_col = next((c for c in df.columns if c.lower().strip() in valid_rev_names), None)
    
    required_cols = ['price_at_period_end', 'market_cap']
    if revenue_col:
        required_cols.append(revenue_col)
        
    if not all(col in df.columns for col in required_cols):
        return 0, None, None    
            
    # Drop rows lacking core mathematical inputs
    df_complete = df.dropna(subset=required_cols).copy()
    if len(df_complete) < min_consecutive:
        return 0, None, None
    
    df_complete = df_complete.drop_duplicates(subset=['period_end'], keep='last')
    df_complete = df_complete.sort_values('period_end')
    
    # Calculate quarter-to-quarter distance (approx 90 days = 1.0)
    df_complete['quarter_diff'] = df_complete['period_end'].diff() / pd.Timedelta(days=90)
    
    max_consecutive, current_consecutive = 1, 1
    best_start = current_start = df_complete.iloc[0]['period_end']
    best_end = df_complete.iloc[0]['period_end']
    
    for i in range(1, len(df_complete)):
        diff = df_complete.iloc[i]['quarter_diff']
        # 0.8 to 1.2 equates to roughly 72 to 108 days
        if 0.8 <= diff <= 1.2:
            current_consecutive += 1
        else:
            if current_consecutive > max_consecutive:
                max_consecutive = current_consecutive
                best_start = current_start
                best_end = df_complete.iloc[i-1]['period_end']
            current_consecutive = 1
            current_start = df_complete.iloc[i]['period_end']
    
    if current_consecutive > max_consecutive:
        max_consecutive = current_consecutive
        best_start = current_start
        best_end = df_complete.iloc[-1]['period_end']
    
    return max_consecutive, best_start, best_end

def process_all_companies() -> Tuple[List[Dict[str, Any]], int, int]:
    """Executes the continuity filter across the entire active cohort."""
    if not CONFIG_FILE.exists():
        logger.error(f"Configuration {CONFIG_FILE} not found.")
        return [], 0, 0

    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    results = []
    deleted_count, kept_count = 0, 0
    
    logger.info("="*70)
    logger.info("CONTINUITY FILTER: ENFORCING MINIMUM 6-QUARTER BOUNDARY")
    logger.info("="*70)
    
    for sector_name, sector_info in config.get('sectors', {}).items():
        for company in sector_info.get('companies', []):
            if company.get('status') != 'active':
                continue
                
            ticker = str(company.get('ticker')).strip()
            file_name = f"{ticker}_raw.csv"
            file_path = DATA_FOLDER / file_name
            
            if not file_path.exists():
                logger.warning(f"[{ticker}] File missing. Flagging for deletion.")
                company.update({'status': 'delete', 'reason': 'Step 08: Missing file'})
                deleted_count += 1
                continue
            
            try:
                df = pd.read_csv(file_path)
                df_clean = clean_dataframe(df)
                
                max_consecutive, start_date, end_date = count_consecutive_quarters(df_clean, min_consecutive=6)
                
                if max_consecutive >= 6:
                    df_clean.to_csv(file_path, index=False, encoding='utf-8-sig')
                    logger.info(f"[{ticker}] KEPT | {max_consecutive} continuous quarters.")
                    kept_count += 1
                else:
                    file_path.unlink()
                    logger.warning(f"[{ticker}] DELETED | Max sequence: {max_consecutive} (< 6).")
                    company.update({'status': 'delete', 'reason': f'Insufficient continuity ({max_consecutive})'})
                    deleted_count += 1
                
            except Exception as e:
                logger.error(f"[{ticker}] Error during processing: {e}")
                company.update({'status': 'delete', 'reason': f'Parse Error: {e}'})
                deleted_count += 1
    
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        f.write("# FOCAL OBJECTS CONFIGURATION (Updated after Continuity Filter)\n\n")
        yaml.dump(config, f, allow_unicode=True, sort_keys=False, default_flow_style=False, indent=2)
        
    return results, deleted_count, kept_count

if __name__ == "__main__":
    process_all_companies()
