#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PM2.5 Slope Mid-point Threshold Analysis
基於斜率區間中位法的閾值分析
方法：找出一階導數的中位值，定義適應區間
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings('ignore')

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Microsoft YaHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

# 群集配色方案 (C1-C5)
CLUSTER_COLORS = {
    '高屏': '#D62728',           # C1 - 紅色
    '苗中彰投': '#FF7F0E',       # C2 - 橘色
    '雲嘉南': '#BCBD22',         # C3 - 橄欖黃
    '北北基桃竹': '#2CA02C',     # C4 - 綠色
    '宜花東': '#1F77B4'          # C5 - 藍色
}

# 目標疾病
TARGET_DISEASES = {
    'Influenza': 'Influenza_regional_adjusted.csv',
    'Allergic rhinitis': 'Allergicrhinitis_regional_adjusted.csv',
    '急性Rhinosinusitis': '急性Rhinosinusitis_regional_adjusted.csv',
    'URI': 'URI_regional_adjusted.csv'
}

# 群集順序（對應 C1-C5）
CLUSTER_ORDER = ['高屏', '苗中彰投', '雲嘉南', '北北基桃竹', '宜花東']


def find_slope_midpoint_thresholds(x_dense, slope_values, fallback_ratio=0.2):
    """
    使用斜率區間中位法找出兩個門檻
    
    Args:
        x_dense: 密集 x 座標
        slope_values: 一階導數值
        fallback_ratio: 備用比例（當主方法失敗時）
    
    Returns:
        tuple: (threshold_1, threshold_2, s_cutoff)
    """
    s_max = np.max(slope_values)
    s_min = np.min(slope_values)
    
    # 主方法：使用中位值
    s_cutoff = (s_max + s_min) / 2
    
    # 找出 slope = s_cutoff 的交點
    # 利用符號變化來找交點
    diff = slope_values - s_cutoff
    sign_changes = np.diff(np.sign(diff))
    crossing_indices = np.where(sign_changes != 0)[0]
    
    # 過濾邊界
    boundary_margin = len(x_dense) // 20
    crossing_indices = crossing_indices[
        (crossing_indices > boundary_margin) & 
        (crossing_indices < len(x_dense) - boundary_margin)
    ]
    
    if len(crossing_indices) >= 2:
        # 找到兩個交點，取第一個和最後一個
        t1_idx = crossing_indices[0]
        t2_idx = crossing_indices[-1]
        t1 = x_dense[t1_idx]
        t2 = x_dense[t2_idx]
        return t1, t2, s_cutoff
    
    # 備用方法：降低切線標準
    s_cutoff_fallback = s_min + fallback_ratio * (s_max - s_min)
    diff = slope_values - s_cutoff_fallback
    sign_changes = np.diff(np.sign(diff))
    crossing_indices = np.where(sign_changes != 0)[0]
    
    crossing_indices = crossing_indices[
        (crossing_indices > boundary_margin) & 
        (crossing_indices < len(x_dense) - boundary_margin)
    ]
    
    if len(crossing_indices) >= 2:
        t1_idx = crossing_indices[0]
        t2_idx = crossing_indices[-1]
        t1 = x_dense[t1_idx]
        t2 = x_dense[t2_idx]
        return t1, t2, s_cutoff_fallback
    
    # 如果仍然找不到，返回 None
    if len(crossing_indices) == 1:
        return x_dense[crossing_indices[0]], None, s_cutoff_fallback
    
    return None, None, s_cutoff


def analyze_cluster_slope_midpoint(df_cluster, cluster_name):
    """
    對單一群集執行斜率區間中位法分析
    
    Returns:
        dict: 包含分析結果的字典
        None: 如果分析失敗
    """
    try:
        # 準備數據
        x = df_cluster['pm2.5'].values
        y = df_cluster['sex_corrected_incidence'].values
        
        # 移除 NaN
        mask = ~(np.isnan(x) | np.isnan(y))
        x = x[mask]
        y = y[mask]
        
        if len(x) < 20:
            print(f"  Warning: {cluster_name} 數據點不足 (n={len(x)})")
            return None
        
        # 排序
        sorted_indices = np.argsort(x)
        x_sorted = x[sorted_indices]
        y_sorted = y[sorted_indices]
        
        # 1. 三次多項式擬合: f(x) = ax^3 + bx^2 + cx + d
        coeffs = np.polyfit(x_sorted, y_sorted, 3)
        a, b, c, d = coeffs
        
        poly_func = np.poly1d(coeffs)
        y_pred = poly_func(x_sorted)
        
        # 2. 生成密集點
        x_dense = np.linspace(x_sorted.min(), x_sorted.max(), 1000)
        y_dense = poly_func(x_dense)
        
        # 3. 計算一階導數: f'(x) = 3ax^2 + 2bx + c
        slope_values = 3*a*x_dense**2 + 2*b*x_dense + c
        
        # 4. 找出門檻
        threshold_1, threshold_2, s_cutoff = find_slope_midpoint_thresholds(x_dense, slope_values)
        
        # 5. 計算 R²
        r2 = r2_score(y_sorted, y_pred)
        
        # 6. 記錄斜率統計
        s_max = np.max(slope_values)
        s_min = np.min(slope_values)
        
        return {
            'cluster': cluster_name,
            'R2': r2,
            'Threshold_1': threshold_1,
            'Threshold_2': threshold_2,
            'S_max': s_max,
            'S_min': s_min,
            'S_cutoff': s_cutoff,
            'x_sorted': x_sorted,
            'y_sorted': y_sorted,
            'x_dense': x_dense,
            'y_dense': y_dense,
            'slope_values': slope_values,
            'coeffs': coeffs
        }
        
    except Exception as e:
        print(f"  Error analyzing {cluster_name}: {e}")
        return None


def plot_disease_by_clusters_slope_midpoint(disease_name, data_path, output_path):
    """
    為單一疾病生成 5 個子圖（每個群集一個）- 斜率區間中位法
    風格匹配 8. Traditional_Threshold_by_Cluster.py
    """
    # 讀取疾病數據
    disease_file = TARGET_DISEASES[disease_name]
    df = pd.read_csv(os.path.join(data_path, disease_file))
    
    # 創建圖形 (1 row × 5 cols)
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(25, 5), sharey=False)
    
    fig.suptitle(f'{disease_name} - Slope Mid-point Threshold Analysis', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # 對每個群集進行分析和繪圖
    for idx, cluster_name in enumerate(CLUSTER_ORDER):
        ax = axes[idx]
        color = CLUSTER_COLORS[cluster_name]
        
        # 過濾該群集的數據
        df_cluster = df[df['Region'] == cluster_name]
        
        if df_cluster.empty:
            ax.text(0.5, 0.5, f'{cluster_name}\n無數據', 
                   ha='center', va='center', fontsize=14, color='gray')
            ax.set_title(f'C{idx+1}: {cluster_name}', fontsize=12, fontweight='bold')
            continue
        
        # 執行分析
        result = analyze_cluster_slope_midpoint(df_cluster, cluster_name)
        
        if result is None:
            ax.text(0.5, 0.5, f'{cluster_name}\n分析失敗', 
                   ha='center', va='center', fontsize=14, color='red')
            ax.set_title(f'C{idx+1}: {cluster_name}', fontsize=12, fontweight='bold')
            continue
        
        # === 繪製散點圖 ===
        ax.scatter(result['x_sorted'], result['y_sorted'], 
                  color='gray', alpha=0.3, s=20, label='原始數據')
        
        # === 繪製擬合曲線（使用群集配色） ===
        ax.plot(result['x_dense'], result['y_dense'], 
               color=color, linewidth=2.5, label='三次擬合')
        
        t1, t2 = result['Threshold_1'], result['Threshold_2']
        
        # === 繪製門檻點和區域 ===
        if t1 is not None and t2 is not None:
            # 標示適應區（綠色背景）
            ax.axvspan(t1, t2, color=color, alpha=0.1, label='適應區', zorder=1)
            
            # 門檻線（虛線）
            ax.axvline(x=t1, color=color, linestyle='--', linewidth=1.5, alpha=0.7, zorder=4)
            ax.axvline(x=t2, color=color, linestyle='--', linewidth=1.5, alpha=0.7, zorder=4)
            
            # 標註門檻值
            y_range = result['y_sorted'].max() - result['y_sorted'].min()
            y_pos_t1 = result['y_sorted'].max() - y_range * 0.05
            y_pos_t2 = result['y_sorted'].max() - y_range * 0.12
            
            ax.text(t1, y_pos_t1, f'T₁={t1:.1f}', 
                   ha='center', va='top', fontsize=9, color=color,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
            ax.text(t2, y_pos_t2, f'T₂={t2:.1f}', 
                   ha='center', va='top', fontsize=9, color=color,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
            
            title_text = f'C{idx+1}: {cluster_name}\n(R²={result["R2"]:.3f})'
            
        elif t1 is not None:
            # 只有一個門檻
            ax.axvline(x=t1, color=color, linestyle='--', linewidth=1.5, alpha=0.7)
            y_range = result['y_sorted'].max() - result['y_sorted'].min()
            y_pos = result['y_sorted'].max() - y_range * 0.05
            ax.text(t1, y_pos, f'T={t1:.1f}', 
                   ha='center', va='top', fontsize=9, color=color,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
            title_text = f'C{idx+1}: {cluster_name}\n(Single Threshold, R²={result["R2"]:.3f})'
        else:
            title_text = f'C{idx+1}: {cluster_name}\n(No Threshold, R²={result["R2"]:.3f})'
        
        # === 設置標題和標籤 ===
        ax.set_title(title_text, fontsize=11, fontweight='bold', color=color)
        ax.set_xlabel('PM2.5 (μg/m³)', fontsize=10)
        if idx == 0:
            ax.set_ylabel('就診率 (‰)', fontsize=10)
        
        # 添加網格
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 只在第一個子圖顯示圖例
        if idx == 0:
            ax.legend(loc='upper left', fontsize=8)
    
    # 調整佈局
    plt.tight_layout()
    
    # 保存圖片
    output_filename = f'Slope_Midpoint_Analysis_{disease_name.replace(" ", "_")}_by_Cluster.png'
    output_filepath = os.path.join(output_path, output_filename)
    plt.savefig(output_filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 已生成: {output_filename}")


def main():
    """主程式"""
    data_path = "./3-2_gender_adjust+pm25_region"
    output_path = "./7_Threshold_Analysis_Cubic"
    os.makedirs(output_path, exist_ok=True)
    
    print("=" * 80)
    print("PM2.5 Slope Mid-point Threshold Analysis")
    print("=" * 80)
    print("方法: 找出一階導數的中位值 S_cutoff = (S_max + S_min) / 2")
    print("門檻: f'(x) = S_cutoff 的兩個根")
    print("=" * 80)
    print(f"數據來源: {data_path}")
    print(f"輸出目錄: {output_path}")
    print(f"目標疾病: {list(TARGET_DISEASES.keys())}")
    print(f"群集數量: {len(CLUSTER_ORDER)}")
    print("=" * 80)
    
    # 對每種疾病進行分析
    for disease_name in TARGET_DISEASES.keys():
        print(f"\n處理疾病: {disease_name}")
        plot_disease_by_clusters_slope_midpoint(disease_name, data_path, output_path)
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()