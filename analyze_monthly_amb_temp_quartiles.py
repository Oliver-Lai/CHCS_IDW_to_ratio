from pathlib import Path
from datetime import date
import pandas as pd

ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / '6_exposure_by_region' / 'factors_weekly_exposure.csv'
OUTPUT_PATH = ROOT / 'monthly_amb_temp_quartiles.csv'


def main():
    df = pd.read_csv(INPUT_PATH, encoding='utf-8-sig')

    # Convert ISO week to a representative date and then to month label
    df['date'] = pd.to_datetime(
        df.apply(lambda row: date.fromisocalendar(int(row['year']), int(row['week']), 1), axis=1)
    )
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month

    # First compute each month mean for each year across all Taiwan
    monthly_by_year = (
        df.groupby(['year', 'month_num'], as_index=False)['AMB_TEMP']
        .mean()
        .rename(columns={'AMB_TEMP': 'avg_amb_temp'})
        .sort_values(['year', 'month_num'])
        .reset_index(drop=True)
    )

    # Then average across the four years for each month of the year
    four_year_avg = (
        monthly_by_year.groupby(['month_num'], as_index=False)['avg_amb_temp']
        .mean()
        .rename(columns={'avg_amb_temp': 'avg_amb_temp_4yr'})
        .sort_values(['month_num'])
        .reset_index(drop=True)
    )
    four_year_avg['month_name'] = four_year_avg['month_num'].apply(lambda m: f'{m}月')

    values = four_year_avg['avg_amb_temp_4yr']
    q1, q2, q3 = values.quantile([0.25, 0.5, 0.75])

    def bucket(x):
        if pd.isna(x):
            return '無資料'
        if x <= q1:
            return '0~Q1'
        if x <= q2:
            return 'Q1~Q2'
        if x <= q3:
            return 'Q2~Q3'
        return 'Q3~'

    four_year_avg['Q1'] = q1
    four_year_avg['Q2'] = q2
    four_year_avg['Q3'] = q3
    four_year_avg['quartile_group'] = four_year_avg['avg_amb_temp_4yr'].apply(bucket)

    four_year_avg.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')

    print(f'已輸出: {OUTPUT_PATH}')
    print('\n全台灣四年平均月份分組結果：')
    for bucket in ['0~Q1', 'Q1~Q2', 'Q2~Q3', 'Q3~']:
        bucket_months = four_year_avg.loc[four_year_avg['quartile_group'] == bucket, ['month_name', 'avg_amb_temp_4yr']]
        if bucket_months.empty:
            continue
        print(f'{bucket}:')
        for _, row in bucket_months.iterrows():
            print(f"  - {row['month_name']}: {row['avg_amb_temp_4yr']:.2f}")


if __name__ == '__main__':
    main()
