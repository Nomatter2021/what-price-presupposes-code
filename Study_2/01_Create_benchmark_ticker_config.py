import requests
from bs4 import BeautifulSoup
import yaml
import time
import logging
import re

# Configure logging for progress tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')

def load_local_sec_mapping(yaml_file):
    """
    Loads the SEC mapping (Ticker -> CIK & Name) from the previously generated YAML file.
    """
    mapping = {}
    try:
        with open(yaml_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            if not data or 'sectors' not in data:
                return mapping
            
            # Flatten the grouped data into a simple dictionary for quick lookup
            for sector_key, sector_info in data['sectors'].items():
                for company in sector_info.get('companies', []):
                    ticker = company.get('ticker')
                    if ticker:
                        mapping[ticker] = {
                            'cik': company.get('cik'),
                            'name': company.get('name')
                        }
    except FileNotFoundError:
        logging.error(f"File not found: {yaml_file}. Please ensure it exists in the same directory.")
    
    return mapping

def auto_generate_sme_benchmark():
    # 1. Load data from local SEC config
    sec_yaml_file = 'SEC_Ticker_config.yaml'
    logging.info(f"1. Loading SEC data from local file: {sec_yaml_file}...")
    sec_mapping = load_local_sec_mapping(sec_yaml_file)
    
    if not sec_mapping:
        logging.error("Failed to load SEC mapping. Stopping execution.")
        return

    logging.info(f"-> Successfully loaded {len(sec_mapping)} tickers for cross-referencing.")

    # 2. Define New Target Sectors based on high-volume SEC data
    # We use lists to allow grouping multiple Finviz sectors into one broad bucket
    FINVIZ_SECTOR_MAPPING = {
        'Healthcare_Pharma': ['sec_healthcare'],
        'Financials_and_Real_Estate': ['sec_financial', 'sec_realestate'],
        'Technology': ['sec_technology']
    }
    
    # Initialize the YAML output structure
    benchmark_config = {"sectors": {}}
    
    # Anti-bot headers
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
    }

    logging.info("\n2. Starting Finviz scraping & SEC cross-referencing for New Sectors...")
    
    for bucket_name, finviz_codes in FINVIZ_SECTOR_MAPPING.items():
        logging.info(f"\n--- Processing Broad Sector: {bucket_name} ---")
        
        # Setup YAML structure for the current bucket
        benchmark_config["sectors"][bucket_name] = {
            "description": f"SME Companies (<$2B) extracted from Finviz ({', '.join(finviz_codes)})",
            "companies": []
        }
        
        matched_count = 0
        
        # Loop through each specific Finviz filter code within the broad bucket
        for finviz_code in finviz_codes:
            if matched_count >= 200:
                break # Already reached the quota for this broad sector
                
            page_index = 1
            logging.info(f"-> Sub-filter: Scanning {finviz_code}...")
            
            # Scrape until we hit the target or run out of pages
            while matched_count < 200:
                url = f"https://finviz.com/screener.ashx?v=111&f=cap_smallunder,{finviz_code}&r={page_index}"
                
                try:
                    resp = requests.get(url, headers=headers, timeout=10)
                    resp.raise_for_status()
                    soup = BeautifulSoup(resp.text, 'html.parser')
                    
                    # Extract all ticker links
                    links = soup.find_all('a', href=re.compile(r'^quote\.ashx\?t='))
                    
                    page_tickers = []
                    for link in links:
                        t = link.text.strip()
                        if t.isupper() and t.isalpha() and 1 <= len(t) <= 5:
                            if t not in page_tickers:
                                page_tickers.append(t)
                    
                    if not page_tickers:
                        logging.warning(f"No more tickers found at page {page_index} for {finviz_code}.")
                        break # Break while loop to move to the next finviz_code (if any)
                        
                    # Cross-reference with SEC data
                    for t in page_tickers:
                        if t in sec_mapping and matched_count < 200:
                            company_data = {
                                'ticker': t,
                                'cik': sec_mapping[t]['cik'],
                                'name': sec_mapping[t]['name'] 
                            }
                            
                            if company_data not in benchmark_config["sectors"][bucket_name]["companies"]:
                                benchmark_config["sectors"][bucket_name]["companies"].append(company_data)
                                matched_count += 1
                                
                    logging.info(f"Page {page_index} ({finviz_code}): Found {len(page_tickers)} tickers -> Total valid matches: {matched_count}/200")
                    
                    page_index += 20 
                    time.sleep(2.0)  
                    
                except Exception as e:
                    logging.error(f"Error occurred while scraping {finviz_code} at page {page_index}: {e}")
                    break

    # 3. Export to Benchmark_config.yaml
    output_file = 'Benchmark_config.yaml'
    logging.info(f"\n3. Exporting data to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as file:
        yaml.dump(benchmark_config, file, default_flow_style=False, sort_keys=False, allow_unicode=True)
        
    logging.info("="*50)
    logging.info(f"✓ COMPLETION: File {output_file} successfully generated.")
    logging.info("Sector Summary:")
    for sector, data in benchmark_config["sectors"].items():
        logging.info(f"- {sector}: {len(data['companies'])} companies")
    logging.info("="*50)

if __name__ == "__main__":
    auto_generate_sme_benchmark()
