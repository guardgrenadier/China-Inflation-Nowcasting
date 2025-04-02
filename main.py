from datetime import date
import pandas as pd
import numpy as np
from backtest import run_backtest, plot_backtest_results
from CPI_data import CPI_data_diff, count_monthly as cpi_count_monthly, cpi_hist
from PPI_data import PPI_data_diff, count_monthly as ppi_count_monthly, ppi_hist

# ------------------------------设置回测范围------------------------------
backtest_range = pd.date_range(start="2022-01-01", end="2024-12-31", freq='ME')
'''
# CPI回测
cpi_results = run_backtest(CPI_data_diff, cpi_hist, backtest_range, cpi_count_monthly,
                           indicator_type='CPI', n_factors=6, alt_method=False)
plot_backtest_results(cpi_results, 'CPI')

# PPI回测
ppi_results = run_backtest(PPI_data_diff, ppi_hist, backtest_range, ppi_count_monthly,
                           indicator_type='PPI', n_factors=6, alt_method=False)
plot_backtest_results(ppi_results, 'PPI')


# ------------------------------另一种模型收敛方法回测------------------------------
# CPI
cpi_results_alt = run_backtest(CPI_data_diff, cpi_hist, backtest_range, cpi_count_monthly,
                               indicator_type='CPI', n_factors=5, alt_method=True)
plot_backtest_results(cpi_results_alt, 'CPI', alt_method=True)
'''
# PPI(效果不佳, n_factors大于3后难以收敛)
ppi_results_alt = run_backtest(PPI_data_diff, ppi_hist, backtest_range, ppi_count_monthly,
                               indicator_type='PPI', n_factors=3, alt_method=True)
plot_backtest_results(ppi_results_alt, 'PPI', alt_method=True)
