# 執行內插法補值
import os
import pandas as pd

input_dir = "1_filtered_output"
output_dir = "3_interpolated_output"
os.makedirs(output_dir, exist_ok=True)

# 需要補值的欄位（不含日期、地區）
value_columns = ["AMB_TEMP", "NO", "NO2", "NOx", "O3", "PM10", "PM2.5", "RH", "SO2"]
skip_threshold = 27  # 超過 27 週缺值不補

# 補值處理函數
def interpolate_file(file_path, output_path):
    try:
        df = pd.read_csv(file_path, encoding="utf-8-sig")

        # 確保日期正確排序
        df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
        df = df.sort_values('日期').reset_index(drop=True)

        df_interp = df.copy()

        for col in value_columns:
            if col not in df_interp.columns:
                continue

            # 確保欄位為數值型別（將無法轉換的字串轉為 NaN）
            df_interp[col] = pd.to_numeric(df_interp[col], errors='coerce')

            # 找出連續缺值區間長度
            null_series = df_interp[col].isna().astype(int)
            groups = (null_series.diff(1) != 0).cumsum()
            max_consec_na = null_series.groupby(groups).sum().max()

            # 若連續缺值超過27筆，不補這個欄位
            if max_consec_na > skip_threshold:
                print(f"⏭️ 跳過補值: {os.path.basename(file_path)} 欄位 {col}，連續缺值達 {max_consec_na} 週")
                continue

            # 否則使用線性補值
            df_interp[col] = df_interp[col].interpolate(method='linear', limit_direction='both')

        # 儲存補值後的資料
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_interp.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"✅ 已補值: {output_path}")

    except Exception as e:
        print(f"❌ 錯誤: {file_path} 無法補值，原因：{e}")

# 遍歷所有檔案並補值
for dirpath, dirnames, filenames in os.walk(input_dir):
    for filename in filenames:
        if filename.endswith(".csv"):
            file_path = os.path.join(dirpath, filename)

            # 保留相對結構
            relative_path = os.path.relpath(file_path, input_dir)
            output_path = os.path.join(output_dir, relative_path)

            interpolate_file(file_path, output_path)
