import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import numpy as np
import os
import urllib.request

font_path = "NotoSansCJKtc-Regular.otf"
if not os.path.exists(font_path):
    print("正在下載中文字體...")
    url = "https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf"
    urllib.request.urlretrieve(url, font_path)

fm.fontManager.addfont(font_path)
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
plt.rcParams['axes.unicode_minus'] = False

input_file = "./7_疾病暴露資料/急性Rhinosinusitis_filtered_with_exposure.csv"
df = pd.read_csv(input_file)

col = 'case_per_capita(‰)'

df['date'] = pd.to_datetime(
    df['year'].astype(str) + '-' + df['week'].astype(str) + '-1',
    format="%G-%V-%w",
    errors='coerce'
)
df['month'] = df['date'].dt.month
fallback_month = np.clip(np.ceil(df['week'] / 4.333), 1, 12)
df['month'] = df['month'].fillna(fallback_month).astype(int)

monthly_data = df.groupby(['region', 'month'])[col].mean().reset_index()

month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

regions_order = ['中彰投', '北北基桃竹苗', '宜花東', '雲嘉南', '高屏']
regions = [r for r in regions_order if r in df['region'].unique()]
colors = sns.color_palette("tab10", len(regions))
region_color_map = dict(zip(regions, colors))

plt.figure(figsize=(12, 7))

for region in regions:
    region_data = monthly_data[monthly_data['region'] == region].sort_values('month')
    plt.plot(
        region_data['month'],
        region_data[col],
        marker='o',
        label=region,
        color=region_color_map[region],
        linewidth=1.5,
        markersize=5
    )

plt.xticks(range(1, 13), month_names)
plt.xlabel('Month')
plt.ylabel('Mean case per capita (‰)')
plt.title('Seasonal variation of weekly acute Rhinosinusitis case per capita', pad=20, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.10),
    ncol=len(regions),
    frameon=False
)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
out_path = "./7_疾病暴露資料/急性Rhinosinusitis_case_per_capita_monthly.png"
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"圖片已儲存至 {out_path}")
