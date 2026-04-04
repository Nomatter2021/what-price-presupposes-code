"""
STEP 05: MARKET CAPITALIZATION CRAWLER
Fetches historical stock prices and shares outstanding to compute Market Cap (V_Price).
Ensures strict temporal alignment with accounting periods to prevent look-ahead bias.

Methodological constraints:
1. Backward Alignment: Fetches the closest price ON or BEFORE the period end.
2. Month-End Synchronization: Aligns period_end to the last day of the month.
3. Cohort Integrity: Only processes entities marked as 'active' in the configuration.
"""

import pandas as pd
import yfinance as yf
import os
import time
import random
import shutil
import yaml
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# ==================== CONFIGURATION & LOGGING ====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

CONFIG_FILE = Path('survey_config.yaml')
DATA_FOLDER = Path('data/raw')
BACKUP_FOLDER = Path('data/backup')

# Throttling to prevent API rate limiting from Yahoo Finance
SLEEP_BETWEEN_COMPANIES = (5, 8) 

def create_backup(file_path: Path) -> None:
    """Creates a timestamped backup before modifying the raw CSV to prevent data loss."""
    BACKUP_FOLDER.mkdir(parents=True, exist_ok=True)
    backup_name = f"{file_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    backup_path = BACKUP_FOLDER / backup_name
    try:
        shutil.copy2(file_path, backup_path)
        logger.debug(f"Backup created: {backup_name}")
    except Exception as e:
        logger.error(f"Backup failed for {file_path.name}: {e}")

# ==================== CORE PROCESSING ====================

def process_company_market_cap(ticker: str, file_type: str) -> Dict[str, Any]:
    """Fetches market data and merges it into the existing quarterly fundamental CSV."""
    file_path = DATA_FOLDER / f"{ticker}_raw.csv"
    stats = {'success': False, 'rows': 0, 'market_cap_calculated': 0}
    
    if not file_path.exists():
        logger.warning(f"File not found: {file_path.name}")
        return stats
    
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            logger.warning(f"Empty data file for {ticker}, skipping.")
            return stats

        # SYNC FIX: Ensure period_end aligns to Month-End for SEC consistency
        df['period_end'] = pd.to_datetime(df['period_end']) + pd.offsets.MonthEnd(0)
        df['period_end'] = df['period_end'].astype('datetime64[ns]')
        stats['rows'] = len(df)
        
        create_backup(file_path)
        
        # Define API fetch window with buffer
        start_date = df['period_end'].min() - pd.Timedelta(days=45)
        end_date = df['period_end'].max() + pd.Timedelta(days=15)
        
        ticker_obj = yf.Ticker(ticker)
        price_history = ticker_obj.history(start=start_date, end=end_date)
        
        if price_history.empty:
            logger.error(f"No price data found for {ticker} (Potential delisting).")
            return stats
        
        # Standardize index timezone
        price_history.index = pd.to_datetime(price_history.index).tz_localize(None).astype('datetime64[ns]')
        
        # Fetch historical Shares Outstanding (Critical for E3/K_Pi ratio integrity)
        shares_df = None
        try:
            shares_history = ticker_obj.get_shares_full(start=start_date, end=end_date)
            if shares_history is not None and not shares_history.empty:
                shares_df = shares_history.to_frame(name='shares_outstanding')
                shares_df.index = pd.to_datetime(shares_df.index).tz_localize(None).astype('datetime64[ns]')
            else:
                curr_shares = ticker_obj.info.get('sharesOutstanding')
                if curr_shares:
                    shares_df = pd.DataFrame({'shares_outstanding': curr_shares}, index=price_history.index)
                    logger.warning(f"Using current shares for historical periods for {ticker} (Inflation risk).")
        except Exception as e:
            logger.warning(f"Could not fetch historical shares for {ticker}: {e}")

        # MERGE LOGIC: 'backward' alignment prevents look-ahead bias
        df = df.sort_values('period_end')
        price_history = price_history.sort_index()
        
        # Merge Price
        df_merged = pd.merge_asof(
            df,
            price_history[['Close']].reset_index(),
            left_on='period_end',
            right_on='Date',
            direction='backward', 
            tolerance=pd.Timedelta('7 days')
        ).drop(columns=['Date'], errors='ignore')
        
        # Merge Shares
        if shares_df is not None:
            shares_df = shares_df.sort_index().reset_index().rename(columns={'index': 'Date'})
            df_merged = pd.merge_asof(
                df_merged.sort_values('period_end'),
                shares_df,
                left_on='period_end',
                right_on='Date',
                direction='backward',
                tolerance=pd.Timedelta('120 days')
            ).drop(columns=['Date'], errors='ignore')

        # Compute V_Price (Market Capitalization)
        if 'Close' in df_merged.columns:
            df_merged.rename(columns={'Close': 'price_at_period_end'}, inplace=True)
            if 'shares_outstanding' in df_merged.columns:
                df_merged['market_cap'] = (df_merged['price_at_period_end'] * df_merged['shares_outstanding']).round(2)
                stats['market_cap_calculated'] = df_merged['market_cap'].notna().sum()
                logger.info(f"✓ Market Cap computed: {stats['market_cap_calculated']}/{len(df_merged)} rows")
        
        # Export updated dataset
        df_merged.to_csv(file_path, index=False, encoding='utf-8-sig')
        stats['success'] = True
        return stats
        
    except Exception as e:
        logger.error(f"Error processing {ticker}: {e}")
        return stats

# ==================== MAIN EXECUTION ====================

def main() -> None:
    logger.info("=" * 70)
    logger.info("MARKET CAP CRAWLER - RESEARCH REPLICATION SUITE")
    logger.info("=" * 70)
    
    if not CONFIG_FILE.exists():
        logger.error(f"Missing input config file: {CONFIG_FILE}")
        return

    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    all_companies = []
    skipped_count = 0
    
    for sector_name, sector_info in config.get('sectors', {}).items():
        for company in sector_info.get('companies', []):
            if company.get('status') != 'active':
                skipped_count += 1
                continue
                
            all_companies.append({
                'Ticker': company.get('ticker'),
                'Type': company.get('type', 'Focal'), 
                'Sector': sector_name
            })
    
    full_df = pd.DataFrame(all_companies)
    
    if full_df.empty:
        logger.warning("No active companies found in the configuration.")
        return
        
    logger.info(f"Total ACTIVE entities to process: {len(full_df)}")
    logger.info(f"Total SKIPPED entities: {skipped_count}")

    for idx, row in full_df.iterrows():
        ticker = row['Ticker'].strip()
        comp_type = row['Type'].strip()
        file_type = comp_type.replace('Sa', '') if comp_type.startswith('Sa') else comp_type
        
        logger.info(f"[{idx+1}/{len(full_df)}] Processing {ticker} | Sector: {row['Sector']}")
        process_company_market_cap(ticker, file_type)
        
        if idx < len(full_df) - 1:
            time.sleep(random.uniform(*SLEEP_BETWEEN_COMPANIES))

    logger.info("=" * 70)
    logger.info("CRAWL COMPLETE - V_PRICE LAYER FINALIZED")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
