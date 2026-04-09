"""
CLEAN RAW SEC DATA (STATISTICAL METHOD - REVISED)
Rule 1: If CSV does not exist -> mark status as 'delete'.
Rule 2: Missing 'Operating_Margin' column -> mark status as 'delete'. (Simulated fallback applied)
Rule 3: Insufficient data (< 4 valid quarters) -> mark status as 'delete'.
Rule 4: Statistical Anomalies -> mark status as 'delete' if:
        - Rule 4A: Mean margin >= 95% (Absurdly high/Fake)
        - Rule 4B: Standard Deviation == 0 (Suspiciously flat data)

Output: Updated Survey_config.yaml and cleaned data/raw/ directory.
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

CONFIG_FILE = Path("Survey_config.yaml")
RAW_DATA_DIR = Path("data/raw")

def clean_raw_data():
    if not CONFIG_FILE.exists():
        logger.error(f"❌ Configuration file {CONFIG_FILE} not found.")
        return

    logger.info(f"Loading configuration from {CONFIG_FILE}...")
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    stats = {
        "total_checked": 0, 
        "kept": 0, 
        "deleted_missing": 0, 
        "deleted_insufficient": 0,
        "deleted_anomaly": 0
    }

    logger.info("="*60)
    logger.info("STARTING RAW DATA CLEANUP (REVISED)")
    logger.info("="*60)

    for sector_name, sector_info in config.get('sectors', {}).items():
        for company in sector_info.get('companies', []):
            ticker = company['ticker']
            
            if company.get('status') != 'active':
                continue

            stats["total_checked"] += 1
            csv_path = RAW_DATA_DIR / sector_name / f"{ticker}_raw.csv"

            # Rule 1
            if not csv_path.exists():
                logger.warning(f"  ✗ {ticker}: Missing CSV file.")
                company['status'] = 'delete'
                company['reason'] = 'Rule 1: CSV file not found'
                stats["deleted_missing"] += 1
                continue

            try:
                df = pd.read_csv(csv_path)
                
                # SỬA LỖI RULE 2 & 3: Mô phỏng logic bù trừ của Step 8 để giữ lại data hợp lệ
                temp_margin = df.get('Operating_Margin', pd.Series([np.nan]*len(df), index=df.index))
                
                if 'Revenue' in df.columns:
                    fallback_margin = (df['Revenue'] - df.get('CostOfRevenue', 0)) / df['Revenue'] - 0.2
                    temp_margin = temp_margin.fillna(fallback_margin)
                
                valid_margins = temp_margin.replace([np.inf, -np.inf], np.nan).dropna()
                valid_count = len(valid_margins)

                # Rule 3 (Đã bao gồm Rule 2): Không đủ 4 quý dữ liệu (kể cả sau khi đã bù trừ)
                if valid_count < 4:
                    logger.warning(f"  ✗ {ticker}: Only {valid_count} quarters (even with fallback).")
                    try:
                        if csv_path.exists(): csv_path.unlink()
                    except Exception as unlink_err:
                        logger.error(f"    - Cannot delete file: {unlink_err}")
                    
                    company['status'] = 'delete'
                    company['reason'] = f'Rule 3: Insufficient data ({valid_count} quarters)'
                    stats["deleted_insufficient"] += 1
                    continue
                
                # Rule 4A & 4B
                mean_margin = valid_margins.mean()
                std_margin = valid_margins.std()
                
                if mean_margin >= 0.95:
                    logger.warning(f"  ✗ {ticker}: Absurd mean margin ({mean_margin:.2f}).")
                    try:
                        if csv_path.exists(): csv_path.unlink()
                    except Exception as unlink_err:
                        logger.error(f"    - Cannot delete file: {unlink_err}")
                        
                    company['status'] = 'delete'
                    company['reason'] = f'Rule 4A: Anomalous mean margin ({mean_margin:.2f})'
                    stats["deleted_anomaly"] += 1
                    continue

                if std_margin == 0:
                    logger.warning(f"  ✗ {ticker}: Zero standard deviation.")
                    try:
                        if csv_path.exists(): csv_path.unlink()
                    except Exception as unlink_err:
                        logger.error(f"    - Cannot delete file: {unlink_err}")
                        
                    company['status'] = 'delete'
                    company['reason'] = 'Rule 4B: Standard deviation is zero (Fake/Flat data)'
                    stats["deleted_anomaly"] += 1
                    continue

                # Pass all
                stats["kept"] += 1
                logger.debug(f"  ✓ {ticker}: Kept ({valid_count} valid quarters)")

            except Exception as e:
                logger.error(f"  ✗ {ticker}: Error reading CSV ({str(e)}).")
                try:
                    if csv_path.exists(): 
                        csv_path.unlink()
                except Exception as unlink_err:
                    logger.error(f"    - Cannot delete file: {unlink_err}")
                
                company['status'] = 'delete'
                company['reason'] = f'Error: {str(e)}'
                stats["deleted_anomaly"] += 1

    logger.info("\nWriting updated statuses back to YAML config...")
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            f.write("# FOCAL OBJECTS CONFIGURATION (Updated after Revised Clean)\n\n")
            try:
                yaml.dump(config, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
            except TypeError:
                # Dự phòng cho PyYAML cũ
                yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
        logger.info(f"  ✓ Đã lưu thành công vào: {CONFIG_FILE.name}")
    except Exception as e:
        logger.error(f"  ❌ Lỗi nghiêm trọng khi lưu file YAML: {e}")

    logger.info("\n" + "="*60)
    logger.info("CLEANUP COMPLETE: STATISTICAL SUMMARY")
    logger.info(f"  - Checked  : {stats['total_checked']}")
    logger.info(f"  - Kept     : {stats['kept']}")
    logger.info(f"  - Deleted  : {stats['total_checked'] - stats['kept']}")
    logger.info("="*60)

if __name__ == "__main__":
    clean_raw_data()
