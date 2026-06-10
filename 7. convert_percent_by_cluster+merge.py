# 將6. inverse_weight2cluster.py的結果轉換為以疾病區分的資料，並以統計出來的就診人數、區域人數計算出就診人數比例
import pandas as pd
import os

# 定義五群與 ID1_CITY 前兩碼對應
region_groups = {
    '北北基桃竹苗': ['01', '31', '11', '32', '33', '35'],  # 台北、新北、基隆、桃園、新竹、苗栗
    '中彰投': ['03', '37', '38'],                         # 台中、彰化、南投
    '雲嘉南': ['39', '40', '05'],                         # 雲林、嘉義、台南
    '高屏': ['07', '43'],                                 # 高雄、屏東
    '宜花東': ['34', '45', '46']                          # 宜蘭、花蓮、台東
}

# 建立反向映射表：城市代碼 → 五群名稱
city_to_region = {}
for region, prefixes in region_groups.items():
    for prefix in prefixes:
        city_to_region[prefix] = region

# 路徑設定
input_folder = "周就醫轉比例"
exposure_path = "./6_exposure_by_region_mean/factors_weekly_exposure.csv"
output_folder = "7_疾病暴露資料_mean"
os.makedirs(output_folder, exist_ok=True)

# 讀取空氣因子暴露資料
exposure_df = pd.read_csv(exposure_path)

# 處理每個疾病檔案
for filename in os.listdir(input_folder):
    if not filename.endswith(".csv"):
        continue

    disease_name = os.path.splitext(filename)[0]
    filepath = os.path.join(input_folder, filename)
    df = pd.read_csv(filepath, dtype={'ID1_CITY': str})

    # 加入 region 欄位
    df["region"] = df["ID1_CITY"].str[:2].map(city_to_region)

    # ❗ 捨棄第 53 週資料
    df = df[df["week"] != 53]

    # 印出未對應的城市代碼
    if df["region"].isnull().any():
        unmatched = df[df["region"].isnull()]["ID1_CITY"].unique()
        print(f"⚠️ {filename} 中有未對應的城市代碼: {', '.join(unmatched)}")

    # 正確做法：每週每區病例與人口都加總（不是分開 group）
    weekly_grouped = df.groupby(["region", "year", "week"]).agg({
        "case_c": "sum",
        "pop_total": "sum"
    }).reset_index()

    # 計算每千人發病比
    weekly_grouped["case_per_capita(‰)"] = (
        weekly_grouped["case_c"] / weekly_grouped["pop_total"] * 1000
    ).round(2)

    # 合併空氣因子
    merged = pd.merge(weekly_grouped, exposure_df, on=["region", "year", "week"], how="left")

    # 輸出
    output_path = os.path.join(output_folder, f"{disease_name}_with_exposure.csv")
    merged.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ 已輸出：{output_path}")
