"""
STEP 01: SEC XBRL DATA CRAWLER
Fetches quarterly financial data for entities defined in 'survey_config.yaml'.
Applies strict discrete quarter filtering (80-105 days) to eliminate Q4 restatement noise, 
preserving the integrity of the R_t metric.
"""

import requests
import pandas as pd
import yaml
import time
import logging
import numpy as np
from pathlib import Path
from requests.exceptions import RequestException
from typing import List, Dict, Optional, Any

# ==================== CONFIGURATION & LOGGING ====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

CONFIG_FILE = Path("survey_config.yaml")
OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEC_USER_AGENT = "Independent Researcher dramainmylife@gmail.com"
SEC_RATE_LIMIT = 0.15  # SEC allows 10 requests/sec max
MAX_RETRIES = 3

XBRL_FALLBACKS = {
    "Revenue": [
        "SalesRevenueNet", "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues", "TotalRevenue", "NetSales"
    ],
    "OperatingIncome": [
        "OperatingIncomeLoss", "ProfitLoss", "OperatingIncome",
        "IncomeLossFromContinuingOperations", "NetIncomeLoss"
    ],
    "CostOfRevenue": [
        "CostOfRevenue", "CostOfGoodsSold", "CostOfSales", "CostOfRevenueAndCostOfGoodsSold"
    ]
}

# ==================== DATA FETCHING ====================

def fetch_sec_metric(cik: str, metric_name: str, fallback_chain: List[str], retries: int = MAX_RETRIES) -> List[Dict[str, Any]]:
    """Fetches financial facts from the SEC API with exponential backoff for rate limits."""
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{str(cik).zfill(10)}.json"
    headers = {"User-Agent": SEC_USER_AGENT}
    
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code == 404:
                return []
                
            response.raise_for_status()
            data = response.json()
            facts = data.get('facts', {}).get('us-gaap', {})
            
            for tag in fallback_chain:
                if tag in facts:
                    units = facts[tag].get('units', {}).get('USD', [])
                    if units:
                        logger.debug(f"Resolved {metric_name} via tag '{tag}'")
                        return units
            return []
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                wait_time = 10 * (attempt + 1)
                logger.warning(f"Rate limited on CIK {cik}. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"HTTP error for {cik}: {e}")
                break
        except RequestException as e:
            logger.error(f"Request failed for {cik}: {e}")
            time.sleep(5)
            
    return []

def parse_quarterly_records(records: List[Dict], metric_name: str) -> pd.DataFrame:
    """Filters for discrete quarters to eliminate cumulative YTD/FY reporting artifacts."""
    if not records:
        return pd.DataFrame()
    
    df = pd.DataFrame(records)
    required = ['start', 'end', 'val', 'fy', 'fp']
    if not all(c in df.columns for c in required):
        return pd.DataFrame()
        
    df = df[required].rename(columns={'start': 'start_date', 'end': 'period_end', 'val': metric_name})
    df['start_date'] = pd.to_datetime(df['start_date'])
    df['period_end'] = pd.to_datetime(df['period_end'])
    df['days'] = (df['period_end'] - df['start_date']).dt.days
    
    # Strict discrete quarter constraint (approx. 3 months)
    df_discrete = df[(df['days'] >= 80) & (df['days'] <= 105)].copy()
    
    # Standardize period_end to MonthEnd to resolve SEC reporting misalignments
    df_discrete['period_end'] = df_discrete['period_end'] + pd.offsets.MonthEnd(0)
    
    # Deduplicate keeping the latest reported amendment
    df_discrete = df_discrete.sort_values(['period_end', metric_name]).drop_duplicates('period_end', keep='last')
    return df_discrete.set_index('period_end')[[metric_name, 'fy', 'fp']]

def calculate_margin(df_rev: pd.DataFrame, df_op: pd.DataFrame, df_cogs: pd.DataFrame) -> pd.Series:
    """Computes operating margin with safe zero-division handling and COGS fallback."""
    margin = pd.Series(index=df_rev.index, dtype=float)
    safe_revenue = df_rev['Revenue'].replace(0, np.nan)
    
    if not df_op.empty and 'OperatingIncome' in df_op.columns:
        margin = df_op['OperatingIncome'] / safe_revenue
    elif not df_cogs.empty and 'CostOfRevenue' in df_cogs.columns:
        margin = (safe_revenue - df_cogs['CostOfRevenue']) / safe_revenue
        
    return margin.clip(-1, 1).astype(float)

# ==================== MAIN EXECUTION ====================

def main() -> None:
    if not CONFIG_FILE.exists():
        logger.error(f"❌ Config file {CONFIG_FILE} not found.")
        return
        
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    logger.info("="*60)
    logger.info("STARTING SEC DATA CRAWL FOR SPECULATIVE REGIME RESEARCH")
    logger.info("="*60)

    for sector_name, sector_info in config.get('sectors', {}).items():
        logger.info(f"\nProcessing Sector: {sector_name}")
        
        for company in sector_info.get('companies', []):
            if company.get('status') != 'active':
                continue
                
            ticker = company['ticker']
            cik = company['cik']
            logger.info(f"  Fetching {ticker} (CIK: {cik})...")
            
            rev_records = fetch_sec_metric(cik, "Revenue", XBRL_FALLBACKS["Revenue"])
            op_records = fetch_sec_metric(cik, "OperatingIncome", XBRL_FALLBACKS["OperatingIncome"])
            cogs_records = fetch_sec_metric(cik, "CostOfRevenue", XBRL_FALLBACKS["CostOfRevenue"])
            
            time.sleep(SEC_RATE_LIMIT)
            
            df_rev = parse_quarterly_records(rev_records, "Revenue")
            df_op = parse_quarterly_records(op_records, "OperatingIncome")
            df_cogs = parse_quarterly_records(cogs_records, "CostOfRevenue")
            
            if df_rev.empty:
                logger.warning(f"  ⊘ {ticker}: No valid discrete quarters found.")
                continue
                
            df_company = df_rev[['Revenue']].copy()
            if not df_op.empty:
                df_company = df_company.join(df_op[['OperatingIncome']], how='left')
            if not df_cogs.empty:
                df_company = df_company.join(df_cogs[['CostOfRevenue']], how='left')
                
            df_company['Operating_Margin'] = calculate_margin(df_rev, df_op, df_cogs)
            
            file_path = OUTPUT_DIR / f"{ticker}_raw.csv"
            df_company.to_csv(file_path)
            
            valid_margins = df_company['Operating_Margin'].notna().sum()
            logger.info(f"  ✓ Saved {ticker}: {len(df_company)} quarters, {valid_margins} valid margins.")

if __name__ == "__main__":
    main()
