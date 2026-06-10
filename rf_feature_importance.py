import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

# 設定字體以支援中文 (如果有中文的話)
plt.rcParams['font.sans-serif'] = ['Taipei Sans TC Beta', 'Microsoft JhengHei', 'PingFang TC', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def main():
    # 1. 讀取資料
    file_path = '7_疾病暴露資料_mean/急性Rhinosinusitis_filtered_with_exposure.csv'
    df = pd.read_csv(file_path)

    # 2. 定義特徵與目標變數
    features = ['AMB_TEMP', 'NO', 'NO2', 'NOx', 'O3', 'PM10', 'PM25', 'RH', 'SO2']
    target = 'case_per_capita(‰)'

    # 移除空值
    df = df.dropna(subset=features + [target])

    X = df[features]
    y = df[target]

    # 3. 建立並訓練隨機森林回歸模型
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    # 4. 取得特徵重要性並排序
    importances = rf.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    # 5. 畫出特徵重要性圖表
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    
    # 根據您的需求自訂 Title
    plt.title('Feature Importance on Acute Rhinosinusitis case_per_capita by Random Forest', fontsize=14)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Factors', fontsize=12)
    plt.tight_layout()

    # 顯示並儲存圖片
    output_img = 'rf_feature_importance.png'
    plt.savefig(output_img, dpi=300)
    print(f"Feature importance plot saved as {output_img}")
    print("\nFeature Importances:")
    print(importance_df.to_string(index=False))

if __name__ == "__main__":
    main()
