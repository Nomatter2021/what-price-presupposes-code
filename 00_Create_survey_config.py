"""
CHUYỂN ĐỔI EXCEL SANG YAML CONFIG (AUTO-ACTIVE)
Input: company_labels.xlsx (File nhập liệu tay, không cần cột Status)
Output: survey_config.yaml (Mặc định gán status: active cho mọi công ty)
"""

import pandas as pd
import yaml
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

INPUT_FILE = Path("company_labels.xlsx")
OUTPUT_FILE = Path("survey_config.yaml")

def clean_value(val):
    """Dọn dẹp dữ liệu trống và chuẩn hóa string"""
    if pd.isna(val):
        return ""
    if isinstance(val, float):
        if val.is_integer():
            return str(int(val))
    return str(val).strip()

def excel_to_yaml():
    if not INPUT_FILE.exists():
        logger.error(f"❌ Không tìm thấy file {INPUT_FILE}.")
        return

    logger.info(f"Đang đọc danh sách thực nghiệm từ {INPUT_FILE}...")
    xls = pd.ExcelFile(INPUT_FILE)
    
    config_dict = {"sectors": {}}
    total_count = 0

    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        if df.empty:
            continue
            
        sector_desc = clean_value(df['Description'].iloc[0]) if 'Description' in df.columns else ""
        companies_list = []
        
        for _, row in df.iterrows():
            ticker = clean_value(row.get('Ticker', ''))
            if not ticker:
                continue
                
            # Chuẩn hóa CIK 10 số
            raw_cik = clean_value(row.get('CIK', '0'))
            cik_formatted = raw_cik.zfill(10) if raw_cik and raw_cik != '0' else '0000000000'
            
            # Tự động gán active theo yêu cầu
            company_data = {
                'ticker': ticker,
                'cik': cik_formatted,
                'status': 'active' 
            }
            
            # Lấy thêm các thông tin bổ trợ nếu có
            if 'Notes' in row and clean_value(row['Notes']):
                company_data['notes'] = clean_value(row['Notes'])
            if 'Reason' in row and clean_value(row['Reason']):
                company_data['reason'] = clean_value(row['Reason'])
            
            companies_list.append(company_data)
            total_count += 1

        if companies_list:
            config_dict["sectors"][sheet_name] = {
                "description": sector_desc,
                "companies": companies_list
            }

    # Xuất ra file YAML
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as file:
        file.write("# CONFIG THỰC NGHIỆM (Tự động gán status: active từ Excel)\n\n")
        yaml.dump(config_dict, file, allow_unicode=True, sort_keys=False, indent=2)

    logger.info(f"✓ Đã tạo {OUTPUT_FILE} | Tổng cộng: {total_count} đối tượng (Trạng thái: active)")

if __name__ == "__main__":
    excel_to_yaml()
