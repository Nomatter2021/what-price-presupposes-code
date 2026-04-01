"""
STEP 2: CRAWL SEC DATA FOR BENCHMARK (MATURE) COHORT
Fetches historical financials for the fixed benchmark groups defined in the research.
Output: data/raw/{Sector}_benchmark_companies.xlsx

Note: The Benchmark Cohort is hardcoded here as these are static reference points 
for the entire empirical study (Methodology Section 4.2).
"""

import requests
import pandas as pd
import time
import logging
from pathlib import Path
from datetime import datetime

# =============================================================================
# CONFIGURATION & LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('benchmark_crawl.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# SEC API Requirements
SEC_USER_AGENT = "Independent Researcher dramainmylife@gmail.com" # PLEASE UPDATE
SEC_RATE_LIMIT = 0.2 # SEC limit is 10 requests/sec; 0.2s is safe
RAW_DATA_DIR = Path("data/benchmark/raw")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# XBRL Tags with prioritized fallbacks for robust cross-sector matching
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

# =============================================================================
# FIXED BENCHMARK COHORT (Hardcoded as per Methodology Section 4.5)
# Mid-Cap peers ($1B-$50B), IPO before 2012 to ensure historical coverage.
# =============================================================================
BENCHMARK_COHORT = {
    "Technology": [
        {"Ticker": "CSGP", "CIK": "0000909832"}, {"Ticker": "VRSN", "CIK": "0001014473"},
        {"Ticker": "FFIV", "CIK": "0001048695"}, {"Ticker": "JNPR", "CIK": "0001043604"},
        {"Ticker": "AKAM", "CIK": "0001013871"}, {"Ticker": "MCHP", "CIK": "0000827054"},
        {"Ticker": "NTAP", "CIK": "0001002047"}, {"Ticker": "STX", "CIK": "0001121788"},
        {"Ticker": "APH", "CIK": "0000820313"}, {"Ticker": "LRCX", "CIK": "0000707549"},
        {"Ticker": "KLAC", "CIK": "0000319201"}, {"Ticker": "MRVL", "CIK": "0001058057"},
        {"Ticker": "MPWR", "CIK": "0001280452"}, {"Ticker": "TER", "CIK": "0000097210"},
        {"Ticker": "CDNS", "CIK": "0000813672"}
    ],
    "Services": [
        {"Ticker": "PAYX", "CIK": "0000723531"}, {"Ticker": "EFX", "CIK": "0000033185"},
        {"Ticker": "FIS", "CIK": "0001136893"}, {"Ticker": "BR", "CIK": "0001383312"},
        {"Ticker": "FISV", "CIK": "0000798354"}, {"Ticker": "JKHY", "CIK": "0000798941"},
        {"Ticker": "ACN", "CIK": "0001467373"}, {"Ticker": "VRSK", "CIK": "0001442145"},
        {"Ticker": "EXPD", "CIK": "0000746515"}, {"Ticker": "CHRW", "CIK": "0001043277"},
        {"Ticker": "ROL", "CIK": "0000084839"}, {"Ticker": "CTAS", "CIK": "0000723254"},
        {"Ticker": "FLT", "CIK": "0001175456"}, {"Ticker": "TYL", "CIK": "0000867057"},
        {"Ticker": "CACI", "CIK": "0000016732"}
    ],
    "Retail": [
        {"Ticker": "TJX", "CIK": "0000109198"}, {"Ticker": "ROST", "CIK": "0000745732"},
        {"Ticker": "TSCO", "CIK": "0000916365"}, {"Ticker": "ULTA", "CIK": "0001403568"},
        {"Ticker": "DKS", "CIK": "0001169551"}, {"Ticker": "GPS", "CIK": "0000039911"},
        {"Ticker": "ANF", "CIK": "0000885639"}, {"Ticker": "URBN", "CIK": "0000892553"},
        {"Ticker": "DECK", "CIK": "0000918947"}, {"Ticker": "DG", "CIK": "0000029534"},
        {"Ticker": "DLTR", "CIK": "0000935703"}, {"Ticker": "BBY", "CIK": "0000764478"},
        {"Ticker": "AEO", "CIK": "0000919012"}, {"Ticker": "KSS", "CIK": "0000885639"},
        {"Ticker": "M", "CIK": "0000794367"}
    ]
}

# =============================================================================
# DATA PROCESSING CORE
# =============================================================================

def format_cik(cik):
    """Normalize CIK to 10-digit string for SEC API compatibility."""
    return str(cik).strip().zfill(10)

def fetch_sec_metric(cik: str, metric_name: str, fallback_chain: list) -> list:
    """Fetch financial facts from SEC API with retry on rate limit."""
    cik_padded = format_cik(cik)
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded}.json"
    headers = {"User-Agent": SEC_USER_AGENT}
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 429:
            logger.warning("Rate limit hit. Sleeping for 10s...")
            time.sleep(10)
            return fetch_sec_metric(cik, metric_name, fallback_chain)
        
        response.raise_for_status()
        data = response.json()
        facts = data.get('facts', {}).get('us-gaap', {})
        
        for tag in fallback_chain:
            if tag in facts:
                units = facts[tag].get('units', {}).get('USD', [])
                if units: return units
        return []
    except Exception as e:
        logger.error(f"Error fetching CIK {cik_padded}: {e}")
        return []

def parse_to_dataframe(records: list, metric_name: str) -> pd.DataFrame:
    """Clean SEC records into quarterly dataframes, filtering for discrete quarters."""
    if not records: return pd.DataFrame()
    df = pd.DataFrame(records)
    
    # Filter for discrete 3-month periods (approx 80-105 days)
    df['start'] = pd.to_datetime(df['start'])
    df['end'] = pd.to_datetime(df['end'])
    df['days'] = (df['end'] - df['start']).dt.days
    df = df[(df['days'] >= 80) & (df['days'] <= 105)].copy()
    
    # Align to MonthEnd to fix SEC date mismatches (e.g., 12-30 vs 12-31)
    df['end'] = df['end'] + pd.offsets.MonthEnd(0)
    df = df.rename(columns={'val': metric_name, 'end': 'period_end'})
    df = df.sort_values(['period_end', metric_name]).drop_duplicates('period_end', keep='last')
    
    return df.set_index('period_end')[[metric_name]]

def crawl_benchmark_data():
    """Execute the crawl for the hardcoded benchmark cohort."""
    logger.info("="*60)
    logger.info("STARTING SEC BENCHMARK DATA CRAWL")
    logger.info("="*60)

    for sector, companies in BENCHMARK_COHORT.items():
        logger.info(f"\n>>> Sector: {sector}")
        excel_file = RAW_DATA_DIR / f"{sector}_benchmark_companies.xlsx"
        sector_payload = {}

        for comp in companies:
            ticker = comp['Ticker']
            logger.info(f"  Fetching {ticker}...")
            
            # Data Acquisition
            rev = parse_to_dataframe(fetch_sec_metric(comp['CIK'], "Revenue", XBRL_FALLBACKS["Revenue"]), "Revenue")
            op = parse_to_dataframe(fetch_sec_metric(comp['CIK'], "OpInc", XBRL_FALLBACKS["OperatingIncome"]), "OperatingIncome")
            cogs = parse_to_dataframe(fetch_sec_metric(comp['CIK'], "COGS", XBRL_FALLBACKS["CostOfRevenue"]), "CostOfRevenue")
            
            if rev.empty: continue
            
            # Merging and Margin Calculation (SAFE DIVISION)
            df = rev.join([op, cogs], how='left')
            safe_rev = df['Revenue'].replace(0, pd.NA)
            
            if 'OperatingIncome' in df.columns:
                df['Operating_Margin'] = (df['OperatingIncome'] / safe_rev).clip(-1, 1)
            elif 'CostOfRevenue' in df.columns:
                df['Operating_Margin'] = ((safe_rev - df['CostOfRevenue']) / safe_rev).clip(-1, 1)
            
            sector_payload[ticker] = df
            time.sleep(SEC_RATE_LIMIT)

        # Save sector-specific Excel with ticker-sheets
        if sector_payload:
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                for t, data in sector_payload.items():
                    data.to_excel(writer, sheet_name=t)
            logger.info(f"✓ Saved sector data to {excel_file}")

if __name__ == "__main__":
    crawl_benchmark_data()
