"""
STEP 10: MERGE ALL VALIDATED CYCLES
Combines all individual company classified files into a single panel dataset 
for cross-sectional and longitudinal statistical analysis.

Input: data/classified/{sector}/{ticker}_classified.csv
Output: data/final_panel.csv
"""

import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

CLASSIFIED_FOLDER = Path('data/classified')
OUTPUT_FILE = Path('data/final_panel.csv')

def merge_all_cycles():
    """Reads all individual CSVs from sector subdirectories and concatenates them."""
    if not CLASSIFIED_FOLDER.exists():
        logger.error(f"❌ Error: Folder {CLASSIFIED_FOLDER} does not exist. Run Step 09 first.")
        return
    
    all_dfs = []
    
    logger.info("="*70)
    logger.info("🔄 MERGING CLASSIFIED DATA INTO PANEL...")
    logger.info("="*70)
    
    # Traverse through sector subdirectories
    for sector_dir in CLASSIFIED_FOLDER.iterdir():
        if not sector_dir.is_dir():
            continue
            
        for filepath in sector_dir.glob("*_classified.csv"):
            ticker = filepath.stem.split('_')[0].upper()
            try:
                df = pd.read_csv(filepath)
                if df.empty:
                    continue
                    
                if 'Ticker' not in df.columns:
                    df['Ticker'] = ticker
                if 'Sector' not in df.columns:
                    df['Sector'] = sector_dir.name
                    
                # Define Cycle_ID as the ticker itself (Step 06 preserved the longest continuous streak)
                df['Cycle_ID'] = ticker 
                df['Source_File'] = filepath.name
                
                all_dfs.append(df)
            except Exception as e:
                logger.error(f"  ⚠️ Error reading {filepath.name}: {e}")
    
    if not all_dfs:
        logger.error("❌ No data available to merge.")
        return
    
    # Concatenate and sort panel data temporally
    final_df = pd.concat(all_dfs, ignore_index=True)
    if 'period_end' in final_df.columns:
        final_df['period_end'] = pd.to_datetime(final_df['period_end'])
        final_df = final_df.sort_values(by=['Ticker', 'period_end']).reset_index(drop=True)
    
    # Export
    final_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    # Summary Statistics
    logger.info("\n" + "="*70)
    logger.info("🎉 MERGE COMPLETE!")
    logger.info(f"📁 Output File : {OUTPUT_FILE}")
    logger.info(f"📊 Total Rows  : {len(final_df):,}")
    logger.info(f"📋 Companies   : {final_df['Ticker'].nunique()}")
    
    if 'Phase' in final_df.columns:
        logger.info("\n📈 Phase Distribution:")
        logger.info("\n" + final_df['Phase'].value_counts().to_string())
    logger.info("="*70)

if __name__ == "__main__":
    merge_all_cycles()
