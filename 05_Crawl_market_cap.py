"""
STEP 7: MARKET CAPITALIZATION CRAWLER
Fetches historical stock prices and shares outstanding to compute Market Cap (V_Price).
Ensures alignment with accounting periods and protects against survival bias.

Methodology Adjustments:
1. Backward Alignment: Fetches the closest price ON or BEFORE the period end.
2. Month-End Synchronization: Aligns period_end to the last day of the month.
3. YAML Integration: ONLY crawls companies with 'status: active' in survey_config.yaml.
"""

import pandas as pd
import yfinance as yf
import os
import time
import random
import shutil
import yaml
from datetime import datetime
from pathlib import Path

# ==================== CONFIGURATION ====================
CONFIG_FILE = 'survey_config.yaml'
DATA_FOLDER = 'data/raw'
BACKUP_FOLDER = 'data/backup'

# Throttling to prevent API rate limiting from Yahoo Finance
SLEEP_BETWEEN_COMPANIES = (5, 8) 

# ==================== LOGGING & AUDIT TRAIL ====================
def log(message, level='INFO'):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    emoji = {'INFO': 'ℹ️', 'SUCCESS': '✅', 'WARNING': '⚠️', 'ERROR': '❌'}
    print(f"[{timestamp}] {emoji.get(level, '•')} {message}")

def create_backup(file_path):
    """Creates a timestamped backup before modifying the raw CSV."""
    if not os.path.exists(BACKUP_FOLDER):
        os.makedirs(BACKUP_FOLDER)
    backup_name = f"{Path(file_path).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    backup_path = os.path.join(BACKUP_FOLDER, backup_name)
    try:
        shutil.copy2(file_path, backup_path)
        log(f"  💾 Backup created: {backup_name}")
    except Exception as e:
        log(f"  Backup failed: {e}", 'ERROR')

# ==================== DATA PROCESSING CORE ====================
def process_company_market_cap(ticker, file_type):
    """Fetches price/shares and merges into the existing quarterly CSV."""
    file_name = f"{ticker}_raw.csv"
    file_path = os.path.join(DATA_FOLDER, file_name)
    
    stats = {'success': False, 'rows': 0, 'market_cap_calculated': 0}
    
    if not os.path.exists(file_path):
        log(f"  ⚠️ File not found: {file_name}", 'WARNING')
        return stats
    
    log(f"📁 Processing: {file_name}")
    
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            log("  Empty file, skipping.", 'WARNING')
            return stats

        # SYNC FIX: Ensure period_end is aligned to Month-End for consistency with Step 03
        df['period_end'] = pd.to_datetime(df['period_end']) + pd.offsets.MonthEnd(0)
        df['period_end'] = df['period_end'].astype('datetime64[ns]')
        stats['rows'] = len(df)
        
        create_backup(file_path)
        
        # Buffer for price fetching
        start_date = df['period_end'].min() - pd.Timedelta(days=45)
        end_date = df['period_end'].max() + pd.Timedelta(days=15)
        
        ticker_obj = yf.Ticker(ticker)
        price_history = ticker_obj.history(start=start_date, end=end_date)
        
        if price_history.empty:
            log(f"  ❌ No price data found for {ticker} (Likely delisted).", 'ERROR')
            return stats
        
        # Align price history index to datetime64[ns]
        price_history.index = pd.to_datetime(price_history.index).tz_localize(None).astype('datetime64[ns]')
        
        # Fetch historical Shares Outstanding (Crucial for E3/K_Pi ratio)
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
                    log("  ⚠️ Using current shares for historical periods (Risk of inflation).", 'WARNING')
        except Exception as e:
            log(f"  ⚠️ Could not fetch shares: {e}", 'WARNING')

        # MERGE LOGIC: Using 'backward' to get the last available price before period end
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

        # Calculate V_Price (Market Cap)
        if 'Close' in df_merged.columns:
            df_merged.rename(columns={'Close': 'price_at_period_end'}, inplace=True)
            if 'shares_outstanding' in df_merged.columns:
                df_merged['market_cap'] = (df_merged['price_at_period_end'] * df_merged['shares_outstanding']).round(2)
                stats['market_cap_calculated'] = df_merged['market_cap'].notna().sum()
                log(f"  ✓ Market Cap computed: {stats['market_cap_calculated']}/{len(df_merged)} rows", 'SUCCESS')
        
        # Save updated CSV
        df_merged.to_csv(file_path, index=False, encoding='utf-8-sig')
        stats['success'] = True
        return stats
        
    except Exception as e:
        log(f"❌ Error processing {ticker}: {str(e)}", 'ERROR')
        return stats

# ==================== MAIN PIPELINE ====================
def main():
    log("=" * 70)
    log("MARKET CAP CRAWLER - RESEARCH REPLICATION SUITE")
    log("=" * 70)
    
    if not os.path.exists(CONFIG_FILE):
        log(f"Missing input config file: {CONFIG_FILE}", 'ERROR')
        return

    # Load Tickers from YAML Config
    log(f"Reading configuration from {CONFIG_FILE}...", 'INFO')
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    all_companies = []
    skipped_count = 0
    
    for sector_name, sector_info in config.get('sectors', {}).items():
        for company in sector_info.get('companies', []):
            
            # ========================================================
            # CHỈ LẤY NHỮNG CÔNG TY CÓ STATUS: ACTIVE
            # ========================================================
            if company.get('status') != 'active':
                log(f"⏭️  Skipping {company.get('ticker')} (Status is not active: {company.get('status')})", 'WARNING')
                skipped_count += 1
                continue
                
            all_companies.append({
                'Ticker': company.get('ticker'),
                'Type': company.get('type', 'Focal'), 
                'Sector': sector_name
            })
    
    full_df = pd.DataFrame(all_companies)
    
    if full_df.empty:
        log("No active companies found in the configuration.", 'WARNING')
        return
        
    log(f"Total ACTIVE companies to process: {len(full_df)}")
    log(f"Total SKIPPED companies: {skipped_count}")

    for idx, row in full_df.iterrows():
        ticker = row['Ticker'].strip()
        comp_type = row['Type'].strip()
        file_type = comp_type.replace('Sa', '') if comp_type.startswith('Sa') else comp_type
        
        log(f"\n[{idx+1}/{len(full_df)}] Ticker: {ticker} | Type: {comp_type} | Sector: {row['Sector']}")
        
        stats = process_company_market_cap(ticker, file_type)
        
        if idx < len(full_df) - 1:
            wait = random.uniform(*SLEEP_BETWEEN_COMPANIES)
            time.sleep(wait)

    log("\n" + "=" * 70)
    log("CRAWL COMPLETE - V_PRICE (MARKET CAP) LAYER READY", 'SUCCESS')
    log("=" * 70)

if __name__ == "__main__":
    main()