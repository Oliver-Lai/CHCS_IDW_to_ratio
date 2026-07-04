#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PM2.5 Slope Mid-point Threshold Analysis - Taiwan-wide (全台)
基於斜率區間中位法的全台分析
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

# 目標疾病與對應檔案
TARGET_DISEASES = {
    'Influenza': 'Influenza_regional_adjusted.csv',
    'Allergic rhinitis': 'Allergicrhinitis_regional_adjusted.csv',
    '急性Rhinosinusitis': '急性Rhinosinusitis_regional_adjusted.csv',
    'URI': 'URI_regional_adjusted.csv'
}

def find_slope_midpoint_thresholds(x_dense, slope_values, fallback_ratio=0.2):
    """
    使用斜率區間中位法找出兩個門檻
    """
    s_max = np.max(slope_values)
    s_min = np.min(slope_values)
    
    # 主方法：使用中位值
    s_cutoff = (s_max + s_min) / 2
    
    # 找出 slope = s_cutoff 的交點
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
    
    if len(crossing_indices) == 1:
        return x_dense[crossing_indices[0]], None, s_cutoff_fallback
    
    return None, None, s_cutoff

def analyze_disease_slope_midpoint(df, disease_name):
    """
    對全台數據執行斜率區間中位法分析
    """
    try:
        # 統一欄位名稱為小寫，避免讀取錯誤
        df.columns = [c.lower() for c in df.columns]
        
        x = df['pm2.5'].values
        y = df['sex_corrected_incidence'].values
        
        # 移除 NaN
        mask = ~(np.isnan(x) | np.isnan(y))
        x = x[mask]
        y = y[mask]
        
        if len(x) < 20:
            return None
        
        # 排序
        sorted_indices = np.argsort(x)
        x_sorted = x[sorted_indices]
        y_sorted = y[sorted_indices]
        
        # 1. 三次多項式擬合
        coeffs = np.polyfit(x_sorted, y_sorted, 3)
        a, b, c, d = coeffs
        poly_func = np.poly1d(coeffs)
        y_pred = poly_func(x_sorted)
        
        # 2. 生成密集點
        x_dense = np.linspace(x_sorted.min(), x_sorted.max(), 1000)
        y_dense = poly_func(x_dense)
        
        # 3. 計算一階導數
        slope_values = 3*a*x_dense**2 + 2*b*x_dense + c
        
        # 4. 找出門檻
        threshold_1, threshold_2, s_cutoff = find_slope_midpoint_thresholds(x_dense, slope_values)
        
        # 5. 計算 R²
        r2 = r2_score(y_sorted, y_pred)
        
        stats = {
            'n_samples': len(x),
            'pm25_min': np.min(x),
            'pm25_max': np.max(x)
        }
        
        return {
            'disease': disease_name,
            'R2': r2,
            'Threshold_1': threshold_1,
            'Threshold_2': threshold_2,
            'S_max': np.max(slope_values),
            'S_min': np.min(slope_values),
            'S_cutoff': s_cutoff,
            'x_sorted': x_sorted,
            'y_sorted': y_sorted,
            'x_dense': x_dense,
            'y_dense': y_dense,
            'slope_values': slope_values,
            'stats': stats
        }
    except Exception as e:
        print(f"  Error analyzing {disease_name}: {e}")
        return None

def plot_taiwan_wide_slope_midpoint(data_path, output_path):
    """
    生成全台分析圖並存檔
    """
    results = []
    
    for disease_name, disease_file in TARGET_DISEASES.items():
        file_path = os.path.join(data_path, disease_file)
        if not os.path.exists(file_path):
            print(f"⚠️ 找不到檔案: {file_path}")
            continue
            
        print(f"正在分析疾病: {disease_name}...")
        df = pd.read_csv(file_path)
        result = analyze_disease_slope_midpoint(df, disease_name)
        
        if result is None:
            continue
        
        results.append(result)
        
        # 繪圖邏輯 (1x2 子圖)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [2, 1]})
        
        # 左圖：數據擬合與區域
        ax1.scatter(result['x_sorted'], result['y_sorted'], color='lightblue', alpha=0.5, s=20, label='Actual Data')
        ax1.plot(result['x_dense'], result['y_dense'], 'r-', linewidth=2, label='Cubic Fit')
        
        t1, t2 = result['Threshold_1'], result['Threshold_2']
        if t1 and t2:
            ax1.axvspan(result['x_dense'].min(), t1, color='#FFE5E5', alpha=0.3, label='Hypersensitive')
            ax1.axvspan(t1, t2, color='#E5F5E5', alpha=0.4, label='Adaptation')
            ax1.axvspan(t2, result['x_dense'].max(), color='#FFE5E5', alpha=0.3, label='Overload')
            ax1.axvline(x=t1, color='black', linestyle='--', alpha=0.7)
            ax1.axvline(x=t2, color='black', linestyle='--', alpha=0.7)
            ax1.text(t1, ax1.get_ylim()[1]*0.9, f'T1={t1:.1f}', rotation=90, ha='right')
            ax1.text(t2, ax1.get_ylim()[1]*0.9, f'T2={t2:.1f}', rotation=90, ha='right')

        ax1.set_title(f'{disease_name} (R²={result["R2"]:.3f})')
        ax1.set_xlabel('PM2.5 (μg/m³)')
        ax1.set_ylabel('Incidence (‰)')
        ax1.legend(loc='best', fontsize=8)
        
        # 右圖：斜率分析
        ax2.plot(result['x_dense'], result['slope_values'], 'g-', label="Slope f'(x)")
        ax2.axhline(y=result['S_cutoff'], color='orange', linestyle='-', label='Cutoff')
        ax2.set_title('Slope Change')
        ax2.set_xlabel('PM2.5')
        ax2.legend(loc='best', fontsize=8)
        
        plt.tight_layout()
        out_name = f'{disease_name.replace(" ", "_")}_slope_analysis.png'
        plt.savefig(os.path.join(output_path, out_name), dpi=300)
        plt.close()
        print(f"  ✅ 已生成: {out_name}")

    # 輸出 CSV 摘要
    if results:
        summary = pd.DataFrame([{
            'Disease': r['disease'], 'R2': r['R2'], 
            'T1': r['Threshold_1'], 'T2': r['Threshold_2'],
            'Samples': r['stats']['n_samples']
        } for r in results])
        summary.to_csv(os.path.join(output_path, 'Taiwan_Wide_Summary.csv'), index=False, encoding='utf-8-sig')

def main():
    """主程式執行區"""
    # 設定路徑
    data_path = "./3-2_gender_adjust+pm25_region"
    output_path = "./7_Threshold_Analysis_Cubic"
    
    # 建立輸出資料夾
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"建立目錄: {output_path}")

    print("=" * 60)
    print("PM2.5 Slope Mid-point Threshold Analysis - START")
    print(f"Input:  {data_path}")
    print(f"Output: {output_path}")
    print("=" * 60)
    
    # 執行繪圖與分析
    plot_taiwan_wide_slope_midpoint(data_path, output_path)
    
    print("\n" + "=" * 60)
    print("分析完成！請檢查輸出資料夾。")
    print("=" * 60)

# 加入啟動開關
if __name__ == "__main__":
    main()