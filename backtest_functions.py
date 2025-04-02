import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime, date
from datetime import timedelta
from DynamicFactorModel import *


def nowcast_cpi(data, n_factors, n_shocks, n_iter):
    """
    通过动态因子模型进行Nowcast的函数
    """
    # 对数据进行线性插值填充nan值
    data_sm = DataInterpolation(data, 0, len(data.index), 'linear').dropna(axis=0, how='any')

    # 动态因子模型
    dfm_em = DFM_EMalgo(data_sm - data_sm.mean(), n_factors, n_shocks, n_iter)

    error = pd.DataFrame(data=np.zeros(shape=(len(data.index), 2)), columns=['shock1', 'shock2'], index=data.index)
    kf = KalmanFilter(Z=data - data_sm.mean(), U=error, A=dfm_em.A, B=dfm_em.B, H=dfm_em.Lambda,
                      state_names=dfm_em.x.columns, x0=dfm_em.x.iloc[0], P0=calculate_covariance(dfm_em.x), Q=dfm_em.Q,
                      R=dfm_em.R)

    predict = RevserseTranslate(kf.x, data_sm.mean(), dfm_em.Lambda, data_sm.columns)

    return predict


def backtest_cpi(data, cpi_history, backtest_res, date=None, types=None, n_mon=None,
                 n_factors =6, n_shocks=2, n_iter=100):
    """
    进行回测的函数，执行了两次nowcast，第一次预测未发布当月值，第二次预测本月CPI
    types = 'CPI' or 'PPI'
    """
    # 第一次循环获取未发布月度数据t月预测值
    predict = nowcast_cpi(data, n_factors, n_shocks, n_iter)

    # 第二次循环，用未发布月度数据t月预测值替代nan值，修正cpi预测值
    # 把data最后一行的月度数据用预测值填上，假设有要预测的月的全部高频数据（t+1月第一天）
    data.iloc[-1, :n_mon] = predict.iloc[-1, :n_mon]

    predict = nowcast_cpi(data, n_factors, n_shocks, n_iter)

    previous_date = pd.Timestamp(date) - pd.offsets.MonthEnd(1)

    # 保存Nowcast的值
    predicted_diff = predict.iloc[-1, 1]
    backtest_res.loc[date, '预测值_差分'] = predicted_diff
    backtest_res.loc[date, '预测值'] = cpi_history.loc[previous_date, types] + predicted_diff


# ------------------------------ 回测结果评价 ------------------------------
def calculate_errors_and_plot(backtest_results, title, column_real='真实值_差分', column_predict='预测值_差分', save_path=None):
    """
    计算误差并绘制误差分布直方图
    """
    errors = backtest_results[column_real] - backtest_results[column_predict]
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=20, edgecolor="black", alpha=0.7)
    plt.title(f"{title}误差分布", fontsize=16)
    plt.xlabel("Error", fontsize=16)
    plt.ylabel("Frequency", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    if save_path:
        plt.savefig(os.path.join('error_analysis_results', save_path))
    plt.show()
    return errors


def calculate_r2(backtest_results, column_real='真实值_差分', column_predict='预测值_差分'):
    """
    计算回测结果的拟合优度 R²
    """
    ss_total = ((backtest_results[column_real] - backtest_results[column_real].mean()) ** 2).sum()
    ss_residual = ((backtest_results[column_real] - backtest_results[column_predict]) ** 2).sum()
    r2 = 1 - (ss_residual / ss_total)
    return r2


def calculate_direction_accuracy(backtest_results, column_real='真实值_差分', column_predict='预测值_差分'):
    """
    计算方向预测正确率
    """
    backtest_results['Direction_Correct'] = (
        ((backtest_results[column_real] > 0) & (backtest_results[column_predict] > 0)) |
        ((backtest_results[column_real] < 0) & (backtest_results[column_predict] < 0)) |
        ((backtest_results[column_real] == 0) & (backtest_results[column_predict].between(-0.05, 0.05)))
    )
    return backtest_results['Direction_Correct'].mean()


def compare_with_wind(backtest_results, column_real, wind_column, title, save_path=None):
    """
    比较万德一致预期与模型预测值的预测效果
    """
    errors = backtest_results[column_real] - backtest_results[wind_column]
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=20, edgecolor="black", alpha=0.7)
    plt.title(f"{title}万德一致预期误差分布", fontsize=16)
    plt.xlabel("Error", fontsize=16)
    plt.ylabel("Frequency", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    if save_path:
        plt.savefig(os.path.join('error_analysis_results', save_path))
    plt.show()
    r2 = calculate_r2(backtest_results, column_real, wind_column)
    print(f"{title}万德一致预期拟合的R²为: {r2}")
    direction_accuracy = calculate_direction_accuracy(backtest_results, column_real, wind_column)
    print(f"{title}万德一致预期方向预测正确率: {direction_accuracy:.2%}")



