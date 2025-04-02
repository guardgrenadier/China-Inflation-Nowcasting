from datetime import date
import pandas as pd
import numpy as np
from Functions import *
import matplotlib.pyplot as plt
import os

file = 'data/PPI_data_raw.xlsx'
data = pd.read_excel(file, parse_dates=True, index_col=0)

# 月度数据和日度周度数据分开
weekly_daily_columns = ['mop_price','HBR_25','South_China','brent','WTI','CRB','LME_AL','cement','copper','indproduct']
monthly_columns = ['PPI','PPI_mop1','PPI_mop2','PPI_mop3','PPI_life','CCPI','export','PMI_raw','PPI_expectation']

# 处理数据
PPI_data_diff = process_data(data, monthly_columns, weekly_daily_columns, end_date='2024-12-31')
count_monthly = count_monthly_indicators(data[monthly_columns])

# 记录下未差分的ppi同比值用于回测画图
ppi_hist = data[['PPI', 'PPI_expectation']]
ppi_hist = ppi_hist.dropna(how='all').resample('ME').last()

os.makedirs('data', exist_ok=True)
#PPI_data_diff.to_csv('data/PPI_data_diff.csv')

