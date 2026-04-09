import yaml
import random
import logging

# Configure logging for better visibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')

def categorize_and_sample(input_file, output_file, sample_size=2000):
    """
    Reads the SIC-grouped YAML, re-categorizes into 3 high-volume main sectors,
    picks random samples, and exports to the final Survey_config format.
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Input file {input_file} not found.")
        return

    # Updated buckets for the 3 most populated broad sectors
    buckets = {
        "Healthcare_Pharma": [],
        "Financials_and_Real_Estate": [],
        "Technology": []
    }

    # Updated keywords based on actual SEC SIC descriptions for the new sectors
    healthcare_keywords = ['pharmaceutical', 'surgical', 'medical', 'health', 'hospital', 'drugs', 'biotech', 'biological', 'diagnostic', 'medicinals']
    finance_re_keywords = ['bank', 'real estate', 'reit', 'insurance', 'finance', 'savings', 'broker', 'investment', 'credit', 'trust', 'commodity']
    tech_keywords = ['technology', 'computer', 'software', 'semiconductor', 'programming', 'data processing', 'electronic']

    logging.info("Categorizing companies into 3 main high-volume sectors...")

    for sic_key, content in data.get('sectors', {}).items():
        desc = content.get('description', '').lower()
        companies = content.get('companies', [])

        target_sector = None
        
        # Keyword matching logic for the new buckets
        if any(kw in desc for kw in healthcare_keywords):
            target_sector = "Healthcare_Pharma"
        elif any(kw in desc for kw in finance_re_keywords):
            target_sector = "Financials_and_Real_Estate"
        elif any(kw in desc for kw in tech_keywords):
            target_sector = "Technology"

        # If a sector matches, add its companies to the respective bucket
        if target_sector:
            buckets[target_sector].extend(companies)

    # Prepare final structure
    survey_config = {"sectors": {}}
    
    # Updated metadata for the final YAML
    sector_meta = {
        "Healthcare_Pharma": "Pharmaceuticals, Medical Devices, and Healthcare Services",
        "Financials_and_Real_Estate": "Commercial Banks, REITs, Insurance, and Financial Services",
        "Technology": "Software, Hardware, Semiconductors, and IT Services"
    }

    logging.info("Starting random sampling...")

    for sector, all_companies in buckets.items():
        # Perform unbiased random sampling
        available_count = len(all_companies)
        actual_sample_size = min(sample_size, available_count)
        
        # Pick random samples if companies exist in the bucket
        if actual_sample_size > 0:
            random_sample = random.sample(all_companies, actual_sample_size)
        else:
            random_sample = []
        
        # Add 'status: active' to each sampled company
        for co in random_sample:
            co['status'] = 'active'

        # Assign to the final config dictionary
        survey_config["sectors"][sector] = {
            "description": sector_meta[sector],
            "companies": random_sample
        }
        
        logging.info(f"Sector {sector}: Sampled {actual_sample_size} out of {available_count} available companies.")

    # Export to YAML
    logging.info(f"Writing final output to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        # Use default_flow_style=False to ensure clean, multi-line YAML formatting
        yaml.dump(survey_config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    logging.info("Done! Survey_config.yaml has been generated successfully.")

if __name__ == "__main__":
    INPUT_FILE = '../SEC_Ticker_config.yaml'
    OUTPUT_FILE = '../Survey_config.yaml'
    SAMPLE_TARGET = 2000
    
    categorize_and_sample(INPUT_FILE, OUTPUT_FILE, SAMPLE_TARGET)
