# 做Kmeans k=5分群，並加入隨機初始化n_init=50
import os
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from shapely.geometry import box
from matplotlib.patches import Patch

# 程式跳出的警告不影響執行
# 1. Could not find the number of physical cores = joblib 嘗試用 wmic 指令查核心數，但新版 Windows 不再預設包含這個工具
# 2. KMeans is known to have a memory leak on Windows with MKL = Windows + MKL + 多執行緒的組合下有已知記憶體洩漏

# 設定中文字體，避免在 Codespaces/Linux 容器中找不到中文字型時報錯
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_prop = fm.FontProperties(fname=font_path) if os.path.exists(font_path) else None

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Noto Sans CJK TC', 'Noto Sans CJK JP', 'Source Han Sans SC', 'Source Han Sans TC', 'DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# === 0️⃣ 輸出資料夾 ===
output_folder = "./8_clustering_result"
os.makedirs(output_folder, exist_ok=True)

# === 1️⃣ 載入資料 ===
default_factors = ["PM25", "NO2", "O3", "PM10", "SO2", "NO", "NOx"]
input_folder = "./6_exposure_by_town"
gml_path = "./TOWN_MOI_1131028.gml"

# 這裡可依需求調整要一起分群的因子
factors = default_factors

# === 2️⃣ 讀取並轉換鄉鎮邊界 ===
taiwan_map = gpd.read_file(gml_path)
taiwan_map = taiwan_map.set_crs("EPSG:3824").to_crs("EPSG:4326")
taiwan_map = taiwan_map.rename(columns={"名稱": "town"})

# 移除離島（澎湖、金門、馬祖）
main_island_bounds = box(119.9, 21.8, 122.1, 25.5)
taiwan_main = taiwan_map[taiwan_map.intersects(main_island_bounds)].copy()

# === 3️⃣ 定義顏色（由高到低） ===
# colors_hex = ["#AA04AA", "#FF0000", "#FFA500", "#FFFF00", "#23B623"]
colors_hex = ["#D62728", "#FF7F0E", "#BCBD22", "#2CA02C", "#1F77B4"]

# === 4️⃣ 逐年執行分群 ===
for year in range(2015, 2020):
    print(f"\n=== 處理 {year} 年 ===")

    # 每鄉鎮 × 每週 × 每因子矩陣
    feature_frames = []
    for factor in factors:
        csv_path = os.path.join(input_folder, f"{factor}_weekly_exposure_with_ID.csv")
        if not os.path.exists(csv_path):
            print(f"⚠️ 找不到 {csv_path}，略過。")
            continue

        factor_df = pd.read_csv(csv_path)
        factor_year = factor_df[factor_df["year"] == year][["ID", "town", "week", factor]].copy()
        if factor_year.empty:
            print(f"⚠️ {year} 年 {factor} 無資料，略過。")
            continue

        factor_pivot = factor_year.pivot_table(index=["ID", "town"], columns="week", values=factor)
        factor_pivot = factor_pivot.reindex(columns=range(1, 53))  # 確保週數一致
        factor_pivot = factor_pivot.apply(lambda row: row.fillna(row.mean()), axis=1)  # 依每鄉鎮平均補值
        factor_pivot.columns = [f"{factor}_week_{w}" for w in factor_pivot.columns]
        feature_frames.append(factor_pivot)

    if not feature_frames:
        print(f"⚠️ {year} 年沒有可用因子資料，略過。")
        continue

    df_features = pd.concat(feature_frames, axis=1)
    df_features = df_features.fillna(df_features.mean(axis=0))

    # === 標準化後做 KMeans 分群 ===
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_features.values)

    kmeans = KMeans(n_clusters=5, n_init=50, random_state=42)
    df_features["cluster"] = kmeans.fit_predict(X_scaled)
    print(f"✅ {year} 分群完成 (k=5, n_init=50, 使用 {len(factors)} 個因子)")

    # === 計算每群平均特徵強度 ===
    cluster_profile = pd.DataFrame(X_scaled, index=df_features.index, columns=df_features.columns[:-1])
    cluster_means = cluster_profile.mean(axis=1).groupby(df_features["cluster"]).mean().sort_values(ascending=False)
    cluster_order = cluster_means.index.tolist()  # 群組由高→低
    cluster_color_map = {cluster: colors_hex[i] for i, cluster in enumerate(cluster_order)}

    # === 重新編號群（高→低） ===
    new_cluster_map = {old: i + 1 for i, old in enumerate(cluster_order)}
    df_features["cluster_ranked"] = df_features["cluster"].map(new_cluster_map)

    # === 輸出各群鄉鎮 ===
    for rank, group_id in enumerate(cluster_order, start=1):
        cluster_df = df_features[df_features["cluster"] == group_id].reset_index()[["ID", "town"]]
        cluster_df.to_csv(f"{output_folder}/air_factor_group_{year}_rank{rank}.csv", index=False, encoding="utf-8-sig")

    # === 合併地理資料 ===
    df_cluster = df_features.reset_index()[["town", "cluster", "cluster_ranked"]]
    map_with_cluster = taiwan_main.merge(df_cluster, on="town", how="inner")

    # === 畫地圖 ===
    fig, ax = plt.subplots(figsize=(6, 9))
    taiwan_main.boundary.plot(ax=ax, color="gray", linewidth=0.3)

    for cluster_id in sorted(map_with_cluster["cluster"].unique()):
        cluster_gdf = map_with_cluster[map_with_cluster["cluster"] == cluster_id]
        cluster_gdf.plot(ax=ax, color=cluster_color_map.get(cluster_id, "#cccccc"), edgecolor="black", linewidth=0.2)

    ax.set_xlim(119.9, 122.1)
    ax.set_ylim(21.8, 25.5)
    ax.set_axis_off()

    # === 標題 ===
    title_text = f"{year} 年台灣空氣因子時序分群 (K=5)"
    if font_prop is not None:
        fig.suptitle(title_text, fontsize=14, y=0.96, ha="center", fontproperties=font_prop)
    else:
        fig.suptitle(title_text, fontsize=14, y=0.96, ha="center")

    # === 自訂圖例 ===
    legend_elements = [
        Patch(facecolor=colors_hex[i], edgecolor='black', label=f"群組 {i+1}：平均 {cluster_means.iloc[i]:.2f}")
        for i in range(5)
    ]
    if font_prop is not None:
        ax.legend(handles=legend_elements,  loc="lower left", fontsize=8, prop=font_prop)
        ax.set_title("", fontproperties=font_prop)
    else:
        ax.legend(handles=legend_elements,  loc="lower left", fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{output_folder}/air_factors_timeseries_cluster_map_{year}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ {year} 地圖輸出完成：air_factors_timeseries_cluster_map_{year}.png")

print("\n🎯 全部年份分群與地圖繪製完成！")
