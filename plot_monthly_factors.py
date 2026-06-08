import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import numpy as np
import os
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

# 讀取合併好的週資料
input_file = "./6_exposure_by_region/factors_weekly_exposure.csv"
if not os.path.exists(input_file):
    print(f"❌ 找不到檔案: {input_file}")
    exit()

df = pd.read_csv(input_file)

# 1. 將 year, week 轉換為 month
# 注意: df['year'] 和 df['week'] 是 ISO 週數，這裡將每週的第一天(星期一)轉成日期，來判斷這個週屬於哪個月
# format="%Y-%W-%w", 其中 %w 置為 1 表示星期一
df['date'] = pd.to_datetime(
    df['year'].astype(str) + '-' + df['week'].astype(str) + '-1', 
    format="%G-%V-%w",  # %G: ISO year, %V: ISO week
    errors='coerce'
)

# 若有轉不出來的（例如某些跨年的邊界情況），用一個粗略算法來算月份（1個月大約4.33週）
df['month'] = df['date'].dt.month
fallback_month = np.clip(np.ceil(df['week'] / 4.333), 1, 12)
df['month'] = df['month'].fillna(fallback_month).astype(int)

# 2. 要畫圖的測項 (扣掉非數值與時間特徵)
factors = [col for col in df.columns if col not in ['region', 'year', 'week', 'date', 'month']]

# 3. 準備輸出資料夾
output_folder = "6_exposure_by_region/plots"
os.makedirs(output_folder, exist_ok=True)

# 月份名稱，用來當 x 軸
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# 定義不同 region 的顏色 
regions = df['region'].unique()
colors = sns.color_palette("tab10", len(regions))
region_color_map = dict(zip(regions, colors))

# 4. 針對每個測項畫圖
for factor in factors:
    plt.figure(figsize=(10, 6))
    
    # 計算該測項在每個月、每個 region 的平均值
    # 以符合「Seasonal variation of weekly concentrations」的形式
    monthly_data = df.groupby(['region', 'month'])[factor].mean().reset_index()
    
    for region in regions:
        region_data = monthly_data[monthly_data['region'] == region]
        region_data = region_data.sort_values('month')
        
        plt.plot(
            region_data['month'], 
            region_data[factor], 
            marker='o', 
            label=region, 
            color=region_color_map[region],
            linewidth=1.5,
            markersize=5
        )
    
    # 設定 x, y 軸
    plt.xticks(range(1, 13), month_names)
    plt.xlabel('Month')
    plt.ylabel(f'Mean {factor}')
    plt.title(f'Seasonal variation of weekly {factor}', pad=20, fontweight='bold')
    
    # 加入 Y 軸的淺色虛線網格線 (模擬附圖)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    
    # 將圖例放在圖表下方
    plt.legend(
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.12), 
        ncol=len(regions), 
        frameon=False # 去掉圖例外框
    )
    
    # 去除上方和右方的邊線
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plot_path = os.path.join(output_folder, f'{factor}_monthly_variation.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

print(f"✅ 所有因素的月平均折線圖已產生並儲存至 {output_folder}/")