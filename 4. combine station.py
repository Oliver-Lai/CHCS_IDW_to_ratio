# 將個別站點資料整合成所有站點合併的年度周資料
# 原本: 二林各空汙因子值1-52周、大里各空汙因子值1-52周...
# 整理後: NO-所有站點1-52周的值、PM25-所有站點1-52周的值...
import os
import pandas as pd
import re

def process_interpolated_files_stationwise():
    input_dir = r'3_interpolated_output'
    output_base_dir = r'4_interpolated_yearly_stationwise'
    os.makedirs(output_base_dir, exist_ok=True)

    # 處理年份範圍
    years = range(2015, 2020)  # 2015~2019

    # 移除"離島"
    areas = ['中部', '北部', '竹苗', '宜蘭', '花東', '高屏', '雲嘉南']

    # 多個測項
    target_factors = ["AMB_TEMP", "NO", "NO2", "NOx", "O3", "PM10", "PM2.5", "RH", "SO2"]
    factor_filename_map = {f: f.replace('.', '') for f in target_factors}  # 處理 PM2.5 -> PM25 檔名用

    for factor in target_factors:
        output_factor_name = factor_filename_map[factor]

        for year in years:
            print(f'\n📅 處理 {year} 年 {factor} 數據...')
            dfs = []

            year_path = os.path.join(input_dir, str(year))
            if not os.path.exists(year_path):
                print(f"⚠️ 路徑不存在：{year_path}")
                continue

            for area in areas:
                area_path = os.path.join(year_path, area)
                if not os.path.exists(area_path):
                    continue

                for filename in os.listdir(area_path):
                    if filename.endswith(".csv"):
                        file_path = os.path.join(area_path, filename)
                        try:
                            df = pd.read_csv(file_path, encoding='utf-8-sig')
                            df['日期'] = pd.to_datetime(df['日期'], errors='coerce')

                            # 從檔名提取測站名稱
                            match = re.match(r"2_周統計_平均值_(.+?)_\d{4}\.csv", filename)
                            if not match:
                                print(f"❓ 無法從檔名提取測站名稱: {filename}")
                                continue
                            station_name = match.group(1)

                            if factor in df.columns:
                                sub_df = df[['日期', factor]].copy()
                                sub_df.rename(columns={factor: station_name}, inplace=True)
                                dfs.append(sub_df)
                        except Exception as e:
                            print(f"❌ 錯誤處理 {file_path}：{e}")

            # 合併所有測站該測項的資料
            if dfs:
                merged_df = dfs[0]
                for df in dfs[1:]:
                    merged_df = pd.merge(merged_df, df, on='日期', how='outer')

                merged_df.sort_values('日期', inplace=True)

                # 儲存資料夾為 interpolated_yearly_stationwise/NO/、PM25/ 等
                factor_output_dir = os.path.join(output_base_dir, output_factor_name)
                os.makedirs(factor_output_dir, exist_ok=True)

                output_file = os.path.join(factor_output_dir, f"{output_factor_name}_merged_{year}.csv")
                merged_df.to_csv(output_file, index=False, encoding="utf-8-sig", na_rep='NaN')
                print(f"✅ 已輸出：{output_file}（共 {len(merged_df)} 筆）")
            else:
                print(f"⏭️ 無資料：{year} 年 {factor}")

if __name__ == "__main__":
    process_interpolated_files_stationwise()
