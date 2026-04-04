"""
STEP 02: RAW SEC DATA CLEANING
Validates the integrity of crawled data.
Rule 1: Identifies and flags missing files.
Rule 2: Detects extreme mathematical anomalies (e.g., cumulative margin > 10).
Updates the global 'survey_config.yaml' state machine accordingly.
"""

import pandas as pd
import yaml
from pathlib import Path
import logging

# ==================== CONFIGURATION ====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

CONFIG_FILE = Path("survey_config.yaml")
RAW_DATA_DIR = Path("data/raw")

def update_yaml_status(config: dict) -> None:
    """Safely rewrites the configuration file with updated entity statuses."""
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        f.write("# FOCAL OBJECTS CONFIGURATION (Updated after Raw Data Clean)\n\n")
        yaml.dump(config, f, allow_unicode=True, sort_keys=False, default_flow_style=False, indent=2)

def main() -> None:
    if not CONFIG_FILE.exists():
        logger.error(f"❌ Configuration file {CONFIG_FILE} not found.")
        return

    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    stats = {"total": 0, "kept": 0, "del_missing": 0, "del_anomaly": 0}
    logger.info("="*60)
    logger.info("STARTING RAW DATA ANOMALY FILTER")
    logger.info("="*60)

    for sector_name, sector_info in config.get('sectors', {}).items():
        for company in sector_info.get('companies', []):
            if company.get('status') != 'active':
                continue

            ticker = company['ticker']
            csv_path = RAW_DATA_DIR / f"{ticker}_raw.csv"
            stats["total"] += 1

            # Rule 1: Missing Entity
            if not csv_path.exists():
                logger.warning(f"  ✗ {ticker}: File missing. Flagged for deletion.")
                company.update({'status': 'delete', 'reason': 'Rule 1: Missing raw CSV'})
                stats["del_missing"] += 1
                continue

            # Rule 2: Anomaly Detection
            try:
                df = pd.read_csv(csv_path, index_col=0)
                if 'Operating_Margin' in df.columns:
                    total_margin = df['Operating_Margin'].sum()
                    if total_margin > 10:
                        logger.warning(f"  ✗ {ticker}: Margin anomaly ({total_margin:.2f} > 10).")
                        csv_path.unlink()
                        company.update({'status': 'delete', 'reason': f'Rule 2: Anomaly ({total_margin:.2f})'})
                        stats["del_anomaly"] += 1
                    else:
                        stats["kept"] += 1
                else:
                    logger.warning(f"  ✗ {ticker}: Missing target metric. Deleting.")
                    csv_path.unlink()
                    company.update({'status': 'delete', 'reason': 'Rule 2: Missing Operating_Margin'})
                    stats["del_anomaly"] += 1
            except Exception as e:
                logger.error(f"  ✗ {ticker}: Parse error ({str(e)}).")
                if csv_path.exists():
                    csv_path.unlink()
                company.update({'status': 'delete', 'reason': f'Error: {str(e)}'})
                stats["del_anomaly"] += 1

    update_yaml_status(config)
    
    logger.info("\n" + "="*60)
    logger.info("CLEANUP COMPLETE: SUMMARY")
    logger.info(f"  - Total Evaluated: {stats['total']}")
    logger.info(f"  - Retained       : {stats['kept']}")
    logger.info(f"  - Dropped (Missing) : {stats['del_missing']}")
    logger.info(f"  - Dropped (Anomaly) : {stats['del_anomaly']}")
    logger.info("="*60)

if __name__ == "__main__":
    main()
