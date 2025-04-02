import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from CPI_data import CPI_data_diff
from backtest_functions import *


# 设置绘图样式
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.use('TkAgg')
os.makedirs('error_analysis_results', exist_ok=True)

# ------------------------------要分析的回测结果------------------------------
indicator_type = 'PPI'  # 'CPI' or 'PPI'
file = f'backtest_results/{indicator_type}_backtest_results.xlsx'
#file = f'backtest_results/{indicator_type}_backtest_results_alt.xlsx'  # 如要评价另一种收敛方法的模型效果

backtest_results = pd.read_excel(file, index_col=0, parse_dates=True)
backtest_range = pd.date_range(start=backtest_results.index[0], end=backtest_results.index[-1], freq='M')  # 回测范围

file2 = 'data/wind_expectation_hist.xlsx'
wind_hist = pd.read_excel(file2, index_col=0, parse_dates=True)
print(wind_hist.head())
# 计算误差并绘制直方图
errors = calculate_errors_and_plot(backtest_results, f'{indicator_type}同比差分预测', save_path=f'{indicator_type}预测误差直方图.jpg')

# 计算 R²
r2 = calculate_r2(backtest_results)
print(f'{indicator_type}一阶差分拟合的R²为: {r2}')

# 计算方向预测正确率
direction_accuracy = calculate_direction_accuracy(backtest_results, '真实值_差分', '预测值_差分')
print(f'{indicator_type}预测方向正确的比例为: {direction_accuracy:.2%}')

# 比较与万德一致预期
backtest_results[f'{indicator_type}万德一致预期_差分'] = backtest_results.index.map(
    lambda date: wind_hist.loc[date, f'{indicator_type}_expectation'] if date in wind_hist.index else None)
compare_with_wind(backtest_results, '真实值_差分', f'{indicator_type}万德一致预期_差分', indicator_type, save_path=f'{indicator_type}万德一致预期')
plot_compare(backtest_results.loc[:, '真实值_差分'], backtest_results.loc[:, f'{indicator_type}万德一致预期_差分'],
             f'{indicator_type}万德一致预期_差分', line_width=2.0, font_size='xx-large', dir='error_analysis_results')