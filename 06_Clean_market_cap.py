"""
STEP 8: MARKET CAP DATA CLEANING & CONTINUITY FILTER
Logic: Scans the entire time series to find the longest continuous sequence.
If no sequence exists with length >= 6 quarters, the file is deleted.
CRITICAL: Updates 'survey_config.yaml' with 'delete' status for failed files 
so downstream scripts know which files to skip.
"""

import pandas as pd
import os
import yaml
from datetime import datetime
from pathlib import Path

# ==================== CONFIGURATION ====================
CONFIG_FILE = 'survey_config.yaml'
# Đảm bảo trỏ đúng thư mục dữ liệu gốc của bạn
DATA_FOLDER = Path('data/raw')

def clean_dataframe(df):
    """Clean dataframe: remove duplicate columns and standardize column names."""
    df = df.loc[:, ~df.columns.duplicated()]
    
    price_cols = [c for c in df.columns if 'price_at_period' in c.lower()]
    if len(price_cols) > 1:
        best_price = max(price_cols, key=lambda c: df[c].notna().sum())
        df = df.drop(columns=[c for c in price_cols if c != best_price])
        df.rename(columns={best_price: 'price_at_period_end'}, inplace=True)
    elif len(price_cols) == 1:
        df.rename(columns={price_cols[0]: 'price_at_period_end'}, inplace=True)
    
    shares_cols = [c for c in df.columns if 'shares_outstanding' in c.lower()]
    if len(shares_cols) > 1:
        best_shares = max(shares_cols, key=lambda c: df[c].notna().sum())
        df = df.drop(columns=[c for c in shares_cols if c != best_shares])
        df.rename(columns={best_shares: 'shares_outstanding'}, inplace=True)
    elif len(shares_cols) == 1:
        df.rename(columns={shares_cols[0]: 'shares_outstanding'}, inplace=True)
    
    if 'price_at_period_end' in df.columns and 'shares_outstanding' in df.columns:
        if 'market_cap' not in df.columns:
            df['market_cap'] = None
        mask = df['market_cap'].isna() & df['shares_outstanding'].notna() & df['price_at_period_end'].notna()
        df.loc[mask, 'market_cap'] = (
            df.loc[mask, 'price_at_period_end'] * df.loc[mask, 'shares_outstanding']
        ).round(2)
    
    return df

def count_consecutive_quarters(df, min_consecutive=6):
    """Count the maximum number of consecutive quarters with full data."""
    if 'period_end' not in df.columns:
        return 0, None, None
    
    df = df.copy()
    df['period_end'] = pd.to_datetime(df['period_end'])
    
    # Chỉ định đích danh các biến thể của cột doanh thu, bỏ qua CostOfRevenue
    valid_rev_names = ['revenue', 'totalrevenue', 'revenues', 'netsales']
    revenue_col = next((c for c in df.columns if c.lower().strip() in valid_rev_names), None)
    
    required_cols = ['price_at_period_end', 'market_cap']
    if revenue_col:
        required_cols.append(revenue_col)
        
    for col in required_cols:
        if col not in df.columns:
            return 0, None, None    
            
    # Lọc dòng có đủ data cốt lõi
    df_complete = df.dropna(subset=required_cols).copy()
    
    if len(df_complete) < min_consecutive:
        return 0, None, None
    
    # Loại bỏ các dòng trùng lặp ngày do bước Sync Month-End tạo ra
    df_complete = df_complete.drop_duplicates(subset=['period_end'], keep='last')
    
    df_complete = df_complete.sort_values('period_end')
    df_complete['quarter_diff'] = df_complete['period_end'].diff() / pd.Timedelta(days=90)
    
    max_consecutive = 1
    current_consecutive = 1
    best_start = df_complete.iloc[0]['period_end']
    best_end = df_complete.iloc[0]['period_end']
    current_start = df_complete.iloc[0]['period_end']
    
    for i in range(1, len(df_complete)):
        diff = df_complete.iloc[i]['quarter_diff']
        
        # 0.8 đến 1.2 tương đương khoảng 72 đến 108 ngày
        if 0.8 <= diff <= 1.2:
            current_consecutive += 1
        else:
            if current_consecutive > max_consecutive:
                max_consecutive = current_consecutive
                best_start = current_start
                best_end = df_complete.iloc[i-1]['period_end']
            current_consecutive = 1
            current_start = df_complete.iloc[i]['period_end']
    
    if current_consecutive > max_consecutive:
        max_consecutive = current_consecutive
        best_start = current_start
        best_end = df_complete.iloc[-1]['period_end']
    
    return max_consecutive, best_start, best_end

def process_all_companies():
    if not os.path.exists(CONFIG_FILE):
        print(f"❌ Error: {CONFIG_FILE} not found.")
        return [], 0, 0

    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("="*90)
    print("🔧 DATA CLEANUP - FILTERING FOR >= 6 CONSECUTIVE QUARTERS")
    print("⚠️  NON-COMPLIANT FILES WILL BE DELETED")
    print("="*90)
    
    results = []
    deleted_count = 0
    kept_count = 0
    
    for sector_name, sector_info in config.get('sectors', {}).items():
        for company in sector_info.get('companies', []):
            if company.get('status') != 'active':
                continue
                
            ticker = str(company.get('ticker')).strip()
            company_type = str(company.get('type', 'Focal')).strip()
            
            file_type = company_type.replace('Sa', '') if company_type.startswith('Sa') else company_type
            file_name = f"{ticker}_raw.csv"
            file_path = DATA_FOLDER / file_name
            
            print(f"{ticker:6} | {file_type:7} | {sector_name:12}", end=" ")
            
            if not file_path.exists():
                print(f"❌ FILE NOT FOUND ({file_path})")
                company['status'] = 'delete'
                company['reason'] = 'Step 08: CSV File Not Found'
                results.append({
                    'ticker': ticker, 'type': file_type, 'sector': sector_name,
                    'file': file_name, 'status': 'NOT_FOUND', 'action': 'DELETED',
                    'max_consecutive_quarters': 0
                })
                deleted_count += 1
                continue
            
            try:
                df = pd.read_csv(file_path)
                original_rows = len(df)
                
                df_clean = clean_dataframe(df)
                
                # CẬP NHẬT: Hạ min_consecutive xuống 6
                max_consecutive, start_date, end_date = count_consecutive_quarters(df_clean, min_consecutive=6)
                
                if max_consecutive >= 6:
                    df_clean.to_csv(file_path, index=False, encoding='utf-8-sig')
                    print(f"✅ KEPT | {max_consecutive:2} quarters | {start_date.date() if start_date else 'N/A'} -> {end_date.date() if end_date else 'N/A'}")
                    kept_count += 1
                    action = 'KEPT'
                else:
                    os.remove(file_path)
                    print(f"❌ DELETED | {max_consecutive:2} quarters (< 6)")
                    company['status'] = 'delete'
                    company['reason'] = f'Step 08: Only {max_consecutive} consecutive quarters (<6)'
                    deleted_count += 1
                    action = 'DELETED'
                
                results.append({
                    'ticker': ticker, 'type': file_type, 'sector': sector_name,
                    'file': file_name, 'status': 'PROCESSED', 'action': action,
                    'original_rows': original_rows, 'max_consecutive_quarters': max_consecutive,
                    'start_date': start_date, 'end_date': end_date
                })
                
            except Exception as e:
                print(f"❌ ERROR: {str(e)[:50]}")
                company['status'] = 'delete'
                company['reason'] = f'Step 08 Error: {str(e)[:50]}'
                results.append({
                    'ticker': ticker, 'type': file_type, 'sector': sector_name,
                    'file': file_name, 'status': 'ERROR', 'action': 'ERROR',
                    'error': str(e)
                })
                deleted_count += 1
    
    print("\n💾 Updating status back to survey_config.yaml...")
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        file_header = "# FOCAL OBJECTS CONFIGURATION (Updated after Step 08 Market Cap Clean)\n\n"
        f.write(file_header)
        yaml.dump(config, f, allow_unicode=True, sort_keys=False, default_flow_style=False, indent=2)
    print("✓ YAML configuration file updated successfully.")

    return results, deleted_count, kept_count

def generate_report(results, deleted_count, kept_count):
    if not results: return
    df_results = pd.DataFrame(results)
    
    print("\n" + "="*90)
    print("📊 SUMMARY REPORT")
    print("="*90)
    print(f"✅ Files kept: {kept_count}")
    print(f"❌ Files deleted/skipped: {deleted_count}")
    print(f"📁 Total processed: {kept_count + deleted_count}")
    
    if kept_count > 0:
        kept_df = df_results[df_results['action'] == 'KEPT']
        print(f"\n🏆 Top 10 kept companies (most consecutive quarters):")
        top10 = kept_df.nlargest(10, 'max_consecutive_quarters')
        for idx, row in top10.iterrows():
            print(f"   {row['ticker']:6} | {row['max_consecutive_quarters']:2} quarters | {row['start_date']} - {row['end_date']}")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = f'data_processing_report_{timestamp}.csv'
    df_results.to_csv(report_file, index=False, encoding='utf-8-sig')
    print(f"\n💾 Report saved: {report_file}")

if __name__ == "__main__":
    print("="*90)
    print("⚠️  WARNING: NON-COMPLIANT FILES (< 6 QUARTERS) WILL BE PERMANENTLY DELETED!")
    print("="*90)
    confirm = input("\nAre you sure you want to proceed? (yes/no): ")
    
    if confirm.lower() != 'yes':
        print("❌ Operation cancelled.")
    else:
        results, deleted_count, kept_count = process_all_companies()
        generate_report(results, deleted_count, kept_count)