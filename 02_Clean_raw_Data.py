"""
CLEAN RAW SEC DATA
Rule 1: If CSV does not exist -> mark status as 'delete'.
Rule 2: If sum of Operating_Margin > 10 -> delete CSV and mark status as 'delete'.
Output: Updated survey_config.yaml and cleaned data/raw/ directory.
"""

import pandas as pd
import yaml
from pathlib import Path
import logging
import os

# Configure logging for reproducible pipeline tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

CONFIG_FILE = Path("survey_config.yaml")
RAW_DATA_DIR = Path("data/raw")

def clean_raw_data():
    """Apply strict cleaning rules to raw CSV files and update config status."""
    if not CONFIG_FILE.exists():
        logger.error(f"❌ Configuration file {CONFIG_FILE} not found.")
        return

    logger.info(f"Loading configuration from {CONFIG_FILE}...")
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    stats = {"total_checked": 0, "kept": 0, "deleted_missing": 0, "deleted_anomaly": 0}

    logger.info("="*60)
    logger.info("STARTING RAW DATA CLEANUP")
    logger.info("="*60)

    # Iterate through the config to check each company
    for sector_name, sector_info in config.get('sectors', {}).items():
        for company in sector_info.get('companies', []):
            # Only process companies that are currently active
            if company.get('status') != 'active':
                continue

            ticker = company['ticker']
            csv_path = RAW_DATA_DIR / f"{ticker}_raw.csv"
            stats["total_checked"] += 1

            # Rule 1: If file does not exist, mark as 'delete'
            if not csv_path.exists():
                logger.warning(f"  ✗ {ticker}: CSV not found. Marking as 'delete'.")
                company['status'] = 'delete'
                company['reason'] = 'Rule 1: Missing raw CSV data'
                stats["deleted_missing"] += 1
                continue

            # Read the CSV to evaluate Rule 2
            try:
                df = pd.read_csv(csv_path, index_col=0)
                
                # Check if 'Operating_Margin' exists and calculate the sum
                if 'Operating_Margin' in df.columns:
                    total_margin = df['Operating_Margin'].sum()
                    
                    # Rule 2: If total margin > 10, delete file and mark as 'delete'
                    if total_margin > 10:
                        logger.warning(f"  ✗ {ticker}: Total margin ({total_margin:.2f}) > 10. Deleting file and marking as 'delete'.")
                        csv_path.unlink() # Physically delete the anomalous CSV
                        
                        company['status'] = 'delete'
                        company['reason'] = f'Rule 2: Anomalous total margin ({total_margin:.2f} > 10)'
                        stats["deleted_anomaly"] += 1
                    else:
                        logger.info(f"  ✓ {ticker}: Data valid. Total margin = {total_margin:.2f}.")
                        stats["kept"] += 1
                else:
                    # If the column is missing entirely, it's equivalent to broken data
                    logger.warning(f"  ✗ {ticker}: 'Operating_Margin' column missing. Deleting file.")
                    csv_path.unlink()
                    company['status'] = 'delete'
                    company['reason'] = 'Rule 2: Missing Operating_Margin column'
                    stats["deleted_anomaly"] += 1

            except Exception as e:
                logger.error(f"  ✗ {ticker}: Error reading CSV ({str(e)}). Marking as 'delete'.")
                if csv_path.exists():
                    csv_path.unlink()
                company['status'] = 'delete'
                company['reason'] = f'Error processing file: {str(e)}'
                stats["deleted_anomaly"] += 1

    # Save the updated statuses back to the YAML configuration file
    logger.info("\nWriting updated statuses back to YAML config...")
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        file_header = "# FOCAL OBJECTS CONFIGURATION (Updated after Raw Data Clean)\n\n"
        f.write(file_header)
        yaml.dump(config, f, allow_unicode=True, sort_keys=False, default_flow_style=False, indent=2)

    # Print final summary
    logger.info("\n" + "="*60)
    logger.info("CLEANUP COMPLETE: SUMMARY")
    logger.info(f"  - Total active companies checked: {stats['total_checked']}")
    logger.info(f"  - Companies kept (Valid data): {stats['kept']}")
    logger.info(f"  - Deleted (Rule 1 - Missing CSV): {stats['deleted_missing']}")
    logger.info(f"  - Deleted (Rule 2 - Anomaly/Sum > 10): {stats['deleted_anomaly']}")
    logger.info("="*60)

if __name__ == "__main__":
    clean_raw_data()
