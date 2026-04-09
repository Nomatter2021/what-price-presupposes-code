import csv
import yaml

def convert_csv_to_yaml(csv_file, yaml_file):
    """
    Reads company data from a CSV file, groups them by SIC code,
    and outputs the result to a YAML configuration file.
    """
    config_data = {"sectors": {}}

    print(f"Reading data from {csv_file}...")
    
    try:
        with open(csv_file, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            for row in reader:
                # 1. Clean the ticker column (e.g., "['AIR']" -> "AIR")
                raw_ticker = row.get('ticker', '').strip()
                clean_ticker = raw_ticker.replace('[', '').replace(']', '').replace("'", "").replace('"', '').strip()
                
                # Skip if the company does not have a valid Ticker (e.g., [])
                if not clean_ticker:
                    continue 

                name = row.get('name', '').strip()
                
                # 2. Format CIK to strictly 10 digits (pad with leading zeros)
                cik = str(row.get('cik', '')).strip().zfill(10)
                sic_code = str(row.get('sic', '')).strip()
                
                # 3. Use 'industry' column for description and remove extra quotes
                sic_desc = row.get('industry', '').strip().replace('"', '')

                # Group by SIC code
                sector_key = f"SIC_{sic_code}"
                if sector_key not in config_data["sectors"]:
                    config_data["sectors"][sector_key] = {
                        "description": sic_desc,
                        "companies": []
                    }

                # Add company to the corresponding sector group
                config_data["sectors"][sector_key]["companies"].append({
                    "ticker": clean_ticker,
                    "cik": cik,
                    "name": name
                })
                
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found. Please check the file path.")
        return

    print("Exporting data to YAML file...")
    with open(yaml_file, 'w', encoding='utf-8') as file:
        yaml.dump(config_data, file, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"Done! Successfully created file: {yaml_file}")

if __name__ == "__main__":
    INPUT_CSV = '../grouped_cik_with_industry.csv'
    OUTPUT_YAML = '../SEC_Ticker_config.yaml'
    
    convert_csv_to_yaml(INPUT_CSV, OUTPUT_YAML)
