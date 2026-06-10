import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from scipy.stats import linregress
import matplotlib.patches as mpatches
import matplotlib as mpl
import os
import urllib.request
import matplotlib.font_manager as fm

font_path = "NotoSansCJKtc-Regular.otf"
if not os.path.exists(font_path):
    print("正在下載中文字體...")
    url = "https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf"
    urllib.request.urlretrieve(url, font_path)

fm.fontManager.addfont(font_path)
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_csv('7_疾病暴露資料_mean/急性Rhinosinusitis_filtered_with_exposure.csv')

# 确定唯一的区域和对应的颜色
regions = df['region'].unique()
colors_map = {
    '高屏': '#A020F0',      # 紫色
    '中彰投': '#FF4444',    # 红色
    '雲嘉南': '#FF9F40',    # 橙色
    '北北基桃竹苗': '#FFFF00',  # 黄色
    '宜花東': '#00AA44'     # 绿色
}

# 暴露变量列表
exposure_vars = ['AMB_TEMP', 'NO', 'NO2', 'NOx', 'O3', 'PM10', 'PM25', 'RH', 'SO2']



# 为每个变量创建图表
for var in exposure_vars:
    fig, ax = plt.subplots(figsize=(12, 10))

    # 计算总体的相关系数（不含NaN）
    valid_data = df.dropna(subset=[var, 'case_per_capita(‰)'])

    if len(valid_data) < 2:
        print(f"Warning: {var} 数据不足")
        plt.close(fig)
        continue

    pearson_corr, _ = pearsonr(valid_data[var], valid_data['case_per_capita(‰)'])
    spearman_corr, _ = spearmanr(valid_data[var], valid_data['case_per_capita(‰)'])

    # 为每个区域绘制散点和趋势线
    for region in regions:
        region_data = df[df['region'] == region].dropna(subset=[var, 'case_per_capita(‰)'])

        if len(region_data) == 0:
            continue

        # 绘制散点
        color = colors_map.get(region, '#000000')
        ax.scatter(region_data[var], region_data['case_per_capita(‰)'],
                   color=color, s=80, alpha=0.6, label=region)

        # 计算趋势线（线性回归）
        if len(region_data) > 1:
            x = region_data[var].values
            y = region_data['case_per_capita(‰)'].values

            # 线性回归
            slope, intercept, r_value, p_value, std_err = linregress(x, y)

            # 绘制趋势线
            x_line = np.array([x.min(), x.max()])
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, color=color, linewidth=2.5, alpha=0.8)

    # 添加相关系数文本框（左上角）
    textstr = f'Total\nPearson: {pearson_corr:.3f}\nSpearman: {spearman_corr:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    # 设置标题和标签
    var_name = var
    xlabel = f'{var} '
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Incidence rate (%o)', fontsize=12)
    ax.set_title(f'Acute Rhinosinusitis: {var_name} Scatter Plot', fontsize=14, pad=20)

    # 创建图例（右下角）
    legend_elements = []
    for region in ['高屏', '中彰投', '雲嘉南', '北北基桃竹苗', '宜花東']:
        if region in regions:
            region_data = df[df['region'] == region].dropna(subset=[var, 'case_per_capita(‰)'])
            if len(region_data) > 1:
                x = region_data[var].values
                y = region_data['case_per_capita(‰)'].values
                slope, intercept, _, _, _ = linregress(x, y)
                label = f'{region} (m={slope:.4f})'
            else:
                label = region
            color = colors_map.get(region, '#000000')
            legend_elements.append(mpatches.Patch(facecolor=color, label=label))

    ax.legend(handles=legend_elements, loc='lower right', fontsize=10,
              title='Region', title_fontsize=10, framealpha=0.95)

    # 设置网格
    ax.grid(True, alpha=0.3, linestyle='--')

    # 调整布局
    plt.tight_layout()

    # 保存图表
    filename = f'Rhinosinusitis_{var}_scatter.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Figure saved as {filename}")
    plt.close(fig)

    # 显示统计信息
    print(f"\n{var} - Total Statistics:")
    print(f"Pearson Correlation: {pearson_corr:.4f}")
    print(f"Spearman Correlation: {spearman_corr:.4f}")

    # 显示每个区域的统计信息
    print(f"{var} - Regional Statistics:")
    for region in ['高屏', '中彰投', '雲嘉南', '北北基桃竹苗', '宜花東']:
        if region in regions:
            region_data = df[df['region'] == region].dropna(subset=[var, 'case_per_capita(‰)'])
            if len(region_data) > 1:
                x = region_data[var].values
                y = region_data['case_per_capita(‰)'].values
                slope, intercept, r_value, _, _ = linregress(x, y)
                pearson, _ = pearsonr(x, y)
                spearman, _ = spearmanr(x, y)
                print(f"  {region}:")
                print(f"    Slope: {slope:.4f}")
                print(f"    Pearson: {pearson:.4f}")
                print(f"    Spearman: {spearman:.4f}")
                print(f"    Data points: {len(region_data)}")

print("\n✓ 所有圖表已生成完成")
