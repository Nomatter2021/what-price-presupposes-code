"""
STEP 03: BENCHMARK COHORT DATA CRAWLER
Acquires historical financials for the fixed mature benchmark groups.
Outputs multi-sheet Excel files mapping the sector-minimum surplus rate.
"""

import requests
import pandas as pd
import time
import logging
from pathlib import Path
from typing import List, Dict, Any

# ==================== CONFIGURATION ====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

SEC_USER_AGENT = "Independent Researcher dramainmylife@gmail.com"
SEC_RATE_LIMIT = 0.2
RAW_DATA_DIR = Path("data/benchmark/raw")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# XBRL Hierarchical Fallbacks
XBRL_FALLBACKS = {
    "Revenue": ["SalesRevenueNet", "RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "TotalRevenue", "NetSales"],
    "OperatingIncome": ["OperatingIncomeLoss", "ProfitLoss", "OperatingIncome", "IncomeLossFromContinuingOperations"],
    "CostOfRevenue": ["CostOfRevenue", "CostOfGoodsSold", "CostOfSales", "CostOfRevenueAndCostOfGoodsSold"]
}

# Static Framework Cohort
BENCHMARK_COHORT = {
    "Technology": [
        {"Ticker": "CSGP", "CIK": "0000909832"}, {"Ticker": "VRSN", "CIK": "0001014473"},
        {"Ticker": "FFIV", "CIK": "0001048695"}, {"Ticker": "JNPR", "CIK": "0001043604"},
        {"Ticker": "AKAM", "CIK": "0001013871"}
    ],
    "Services": [
        {"Ticker": "PAYX", "CIK": "0000723531"}, {"Ticker": "EFX", "CIK": "0000033185"},
        {"Ticker": "FIS", "CIK": "0001136893"}, {"Ticker": "BR", "CIK": "0001383312"},
        {"Ticker": "FISV", "CIK": "0000798354"}
    ],
    "Retail": [
        {"Ticker": "TJX", "CIK": "0000109198"}, {"Ticker": "ROST", "CIK": "0000745732"},
        {"Ticker": "TSCO", "CIK": "0000916365"}, {"Ticker": "ULTA", "CIK": "0001403568"},
        {"Ticker": "DKS", "CIK": "0001169551"}
    ]
}

# ==================== DATA PROCESSING ====================

def fetch_sec_metric(cik: str, metric_name: str, fallback_chain: List[str]) -> List[Dict[str, Any]]:
    """Fetches SEC data with integrated rate-limit backoff."""
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{str(cik).zfill(10)}.json"
    headers = {"User-Agent": SEC_USER_AGENT}
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 429:
            logger.warning("Rate limit hit. Initiating backoff (10s)...")
            time.sleep(10)
            return fetch_sec_metric(cik, metric_name, fallback_chain)
        
        response.raise_for_status()
        facts = response.json().get('facts', {}).get('us-gaap', {})
        
        for tag in fallback_chain:
            units = facts.get(tag, {}).get('units', {}).get('USD', [])
            if units: return units
        return []
    except Exception as e:
        logger.error(f"Error fetching CIK {cik}: {e}")
        return []

def parse_to_dataframe(records: List[Dict], metric_name: str) -> pd.DataFrame:
    """Standardizes records into contiguous analytical quarters."""
    if not records: return pd.DataFrame()
    df = pd.DataFrame(records)
    
    df['start'] = pd.to_datetime(df['start'])
    df['end'] = pd.to_datetime(df['end'])
    df['days'] = (df['end'] - df['start']).dt.days
    df = df[(df['days'] >= 80) & (df['days'] <= 105)].copy()
    
    df['end'] = df['end'] + pd.offsets.MonthEnd(0)
    df = df.rename(columns={'val': metric_name, 'end': 'period_end'})
    df = df.sort_values(['period_end', metric_name]).drop_duplicates('period_end', keep='last')
    
    return df.set_index('period_end')[[metric_name]]

def main() -> None:
    logger.info("="*60)
    logger.info("STARTING SEC BENCHMARK COHORT DATA ACQUISITION")
    logger.info("="*60)

    for sector, companies in BENCHMARK_COHORT.items():
        logger.info(f"\n>>> Compiling Benchmark Sector: {sector}")
        excel_file = RAW_DATA_DIR / f"{sector}_benchmark_companies.xlsx"
        sector_payload = {}

        for comp in companies:
            ticker = comp['Ticker']
            logger.info(f"  Fetching {ticker}...")
            
            rev = parse_to_dataframe(fetch_sec_metric(comp['CIK'], "Revenue", XBRL_FALLBACKS["Revenue"]), "Revenue")
            op = parse_to_dataframe(fetch_sec_metric(comp['CIK'], "OpInc", XBRL_FALLBACKS["OperatingIncome"]), "OperatingIncome")
            cogs = parse_to_dataframe(fetch_sec_metric(comp['CIK'], "COGS", XBRL_FALLBACKS["CostOfRevenue"]), "CostOfRevenue")
            
            if rev.empty: continue
            
            df = rev.join([op, cogs], how='left')
            safe_rev = df['Revenue'].replace(0, pd.NA)
            
            if 'OperatingIncome' in df.columns:
                df['Operating_Margin'] = (df['OperatingIncome'] / safe_rev).clip(-1, 1)
            elif 'CostOfRevenue' in df.columns:
                df['Operating_Margin'] = ((safe_rev - df['CostOfRevenue']) / safe_rev).clip(-1, 1)
            
            sector_payload[ticker] = df
            time.sleep(SEC_RATE_LIMIT)

        if sector_payload:
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                for t, data in sector_payload.items():
                    data.to_excel(writer, sheet_name=t)
            logger.info(f"✓ Output finalized for {sector}: {excel_file}")

if __name__ == "__main__":
    main()
