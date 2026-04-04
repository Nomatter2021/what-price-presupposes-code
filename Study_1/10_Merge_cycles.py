"""
STEP 10: MERGE ALL VALIDATED CYCLES
Combines all individual company cycle files into a single panel dataset 
for cross-sectional and longitudinal statistical analysis.
"""

import pandas as pd
import os
from pathlib import Path

# ==================== CONFIGURATION & PATHS ====================
FINAL_FOLDER = Path('data/classified')
OUTPUT_FILE = Path('data/final_all_cycles_combined.csv')


# ==================== MAIN EXECUTION ====================

def merge_all_cycles() -> None:
    """Reads all individual CSVs and concatenates them into a master panel."""
    if not FINAL_FOLDER.exists():
        print(f"❌ Error: Target directory {FINAL_FOLDER} does not exist.")
        return
    
    # Exclude the output file itself to prevent recursive duplication
    files = [f for f in os.listdir(FINAL_FOLDER) 
             if f.endswith('.csv') and f != OUTPUT_FILE.name]
    
    if not files:
        print(f"❌ No valid CSV files found in {FINAL_FOLDER}")
        return
    
    print("="*70)
    print(f"🔄 MERGING {len(files)} CYCLE FILES...")
    print("="*70)
    
    all_dfs = []
    
    for file in files:
        ticker = file.split('_')[0].upper()
        filepath = FINAL_FOLDER / file
        
        try:
            df = pd.read_csv(filepath)
            
            if 'Ticker' not in df.columns:
                df['Ticker'] = ticker
                
            df['Source_File'] = file
            all_dfs.append(df)
            print(f"  ✓ Read: {file:25} | {len(df):4} rows ({ticker})")
            
        except Exception as e:
            print(f"  ⚠️ Error reading {file}: {e}")
    
    if not all_dfs:
        print("❌ No data available to merge.")
        return
    
    # Concatenate and sort panel data temporally
    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df = final_df.sort_values(by=['Ticker', 'period_end']).reset_index(drop=True)
    
    # Export final master dataset
    final_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    # Summary Statistics Output
    print("\n" + "="*70)
    print("🎉 MERGE COMPLETE!")
    print(f"📁 Output File : {OUTPUT_FILE}")
    print(f"📊 Total Rows  : {len(final_df):,}")
    print(f"📋 Companies   : {final_df['Ticker'].nunique()}")
    
    if 'Cycle_Name' in final_df.columns:
        print(f"🔢 Total Cycles: {final_df['Cycle_Name'].nunique()}")
        
    if 'Phase' in final_df.columns:
        print("\n📈 Phase Distribution:")
        print(final_df['Phase'].value_counts().to_string())
    print("="*70)

if __name__ == "__main__":
    merge_all_cycles()
