from datetime import date
import pandas as pd
import numpy as np
from Functions import *
import matplotlib.pyplot as plt
import os

file = 'data/CPI_data_raw.xlsx'
data = pd.read_excel(file, parse_dates=True, index_col=0)

# 月度数据和日度周度数据分开
weekly_daily_columns = ['hainan','gas93#','gas0#','brent','commodity','pork','beef','vegetables','fruits','egg']
monthly_columns = ['import','CPI','CPI_food','CPI_housing','CPI_transcomm','CPI_culture','housing_price','PMI_business','CPI_expectation']

# 处理数据
CPI_data_diff = process_data(data, monthly_columns, weekly_daily_columns, end_date='2024-12-31')
count_monthly = count_monthly_indicators(data[monthly_columns])

# 记录下未差分的cpi同比值用于回测画图
cpi_hist = data[['CPI', 'CPI_expectation']]
cpi_hist = cpi_hist.dropna(how='all').resample('ME').last()

os.makedirs('data', exist_ok=True)
#CPI_data_diff.to_csv('data/CPI_data_diff.csv')