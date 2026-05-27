# 從歷年的周統計資料中篩選出2016-2019的資料，以及需要的空汙因子，以利後續搭配健保資料進行分析
import os
import pandas as pd

root_dir = "周統計資料_獨立"
output_dir = "1_filtered_output"
os.makedirs(output_dir, exist_ok=True)

target_years = ['2015', '2016', '2017', '2018', '2019']
columns_to_keep = ["日期", "地區", "AMB_TEMP","NO", "NO2", "NOx", "O3", "PM10", "PM2.5", "RH","SO2"]

for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.endswith(".csv") and "平均值" in filename:
            for year in target_years:
                if year in filename:
                    file_path = os.path.join(dirpath, filename)
                    try:
                        # 嘗試不同編碼讀取
                        try:
                            df = pd.read_csv(file_path, encoding="utf-8-sig")
                        except UnicodeDecodeError:
                            df = pd.read_csv(file_path, encoding="cp950")

                        # 只保留需要欄位，並避免 SettingWithCopyWarning
                        df_filtered = df.loc[:, columns_to_keep].copy()

                        # 四捨五入數值欄位
                        numeric_cols = df_filtered.columns.difference(['日期', '地區'])
                        df_filtered.loc[:, numeric_cols] = df_filtered[numeric_cols].round(2)

                        # 處理相對路徑，移除第四層結構（保留前兩層）
                        relative_path = os.path.relpath(dirpath, root_dir)
                        parts = relative_path.split(os.sep)
                        trimmed_path = os.path.join(*parts[:2]) if len(parts) >= 2 else relative_path

                        # 建立輸出目錄
                        output_subdir = os.path.join(output_dir, trimmed_path)
                        os.makedirs(output_subdir, exist_ok=True)

                        # 儲存檔案
                        output_path = os.path.join(output_subdir, filename)
                        df_filtered.to_csv(output_path, index=False, encoding="utf-8-sig")
                        print(f"✅ 已處理: {output_path}")
                    except Exception as e:
                        print(f"❌ 錯誤: 無法處理 {file_path}，原因：{e}")
                    break
