import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 讀取現有數據
df = pd.read_csv('data/raw/data.csv')
df['date'] = pd.to_datetime(df['date'])

# 生成新的日期範圍
start_date = df['date'].min()
end_date = start_date + timedelta(days=180)  # 6個月
new_dates = pd.date_range(start=start_date, end=end_date, freq='D')

# 使用現有數據的統計特性來生成新數據
mean_power = df['power_consumption'].mean()
std_power = df['power_consumption'].std()
trend = np.linspace(0, 500, len(new_dates))  # 添加一個緩慢上升的趨勢
seasonality = 100 * np.sin(np.linspace(0, 6*np.pi, len(new_dates)))  # 季節性波動
noise = np.random.normal(0, 50, len(new_dates))  # 隨機噪聲

# 生成新的電力消耗數據
new_power = mean_power + trend + seasonality + noise

# 創建新的DataFrame
new_df = pd.DataFrame({
    'date': new_dates,
    'power_consumption': new_power
})

# 保存到CSV文件
new_df.to_csv('data/raw/extended_data.csv', index=False) 