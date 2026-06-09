# 繪製2016年第45周的PM2.5暴露情況做確認
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box
import matplotlib.font_manager as fm
import os

import urllib
import urllib.request
# ==========================================
# 終極解法：直接下載並載入開源中文字體檔 (更新有效網址)
# ==========================================
font_path = "NotoSansCJKtc-Regular.otf"

# 如果資料夾內沒有字體檔，就自動從 GitHub 下載
if not os.path.exists(font_path):
    print("⬇️ 正在下載中文字體 (Noto Sans CJK TC)...")
    # 使用目前官方最新的有效下載連結
    url = "https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf"
    urllib.request.urlretrieve(url, font_path)
    print("✅ 字體下載完成！")

# 強制將該字體加入 Matplotlib 的字體庫中
fm.fontManager.addfont(font_path)

# 取得該字體在 Matplotlib 系統中的正確名稱，並設為預設
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
plt.rcParams['axes.unicode_minus'] = False # 解決負號無法正常顯示的問題
# ==========================================

# === 1️⃣ 讀取資料 ===
csv_path = "./6_exposure_by_town/PM25_weekly_exposure_with_ID.csv"
gml_path = "./TOWN_MOI_1131028.gml"

df = pd.read_csv(csv_path)
df_2016w45 = df[(df["year"] == 2016) & (df["week"] == 45) & (df["PM25"].notna())]
print(f"共有 {len(df_2016w45)} 個鄉鎮有有效資料")

# === 2️⃣ 讀取鄉鎮邊界 ===
taiwan_map = gpd.read_file(gml_path)
taiwan_map = taiwan_map.set_crs("EPSG:3824").to_crs("EPSG:4326")
taiwan_map = taiwan_map.rename(columns={"名稱": "town"})

# === 3️⃣ 只保留台灣本島 ===
# 定義台灣主島的經緯度邊界（排除澎湖、金門、馬祖）
main_island_bounds = box(119.9, 21.8, 122.1, 25.5)
taiwan_main = taiwan_map[taiwan_map.intersects(main_island_bounds)].copy()

# === 4️⃣ 合併空間資料 ===
map_with_data = taiwan_main.merge(df_2016w45, on="town", how="inner")

# === 5️⃣ 畫地圖 ===
fig, ax = plt.subplots(figsize=(6, 9))

# 底圖邊界
taiwan_main.boundary.plot(ax=ax, linewidth=0.3, color="lightgray")

# 資料圖層
map_with_data.plot(
    column="PM25",
    ax=ax,
    legend=True,
    cmap="OrRd",
    linewidth=0.2,
    edgecolor="black",
    legend_kwds={
        "label": "PM2.5 (μg/m³)",
        "orientation": "vertical",
        "shrink": 0.6,
        "pad": 0.02
    }
)

# === 6️⃣ 美化 ===
ax.set_xlim(119.9, 122.1)
ax.set_ylim(21.8, 25.5)
ax.set_axis_off()

# 標題置中
fig.suptitle("2016 年第 45 週 台灣本島鄉鎮 PM2.5 暴露量", fontsize=14, y=0.95, ha="center")

plt.tight_layout()

# === 7️⃣ 儲存圖片 ===
plt.savefig("./6_exposure_by_town/PM25_2016-45.png", dpi=300, bbox_inches="tight")
