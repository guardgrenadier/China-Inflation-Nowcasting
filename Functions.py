from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime, date
from datetime import timedelta
from statsmodels.tsa.api import VAR
import scipy
import sklearn
import statsmodels.api as sm


def plot_compare(history, forecast, title, line_width=3.0, font_size='xx-large', dir=None):
    """
    画图的函数
    """
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['lines.markersize'] = 6
    plt.rcParams['image.cmap'] = 'gray'

    plt.figure()
    # 添加 observed 数据，带空心小圆圈
    plt.plot(history.index, history, color='r', label='真实值', linewidth=line_width,
             marker='o', markerfacecolor='none', markersize=8)

    # 添加 predicted 数据，带空心小圆圈
    plt.plot(forecast.index, forecast, color='k', label='预测值', linewidth=line_width,
             marker='o', markerfacecolor='none', markersize=8)
    plt.legend()
    plt.xlabel('Date')
    plt.title(title, fontweight='bold', fontsize=font_size)  # 标题加粗
    # 添加 y=0 的实线
    plt.axhline(0, color='black', linestyle='-', linewidth=1, label='diff = 0')

    plt.savefig(os.path.join(dir, f"{title}.jpg"))
    plt.show()

    return


def DataInterpolation(data, start, end, method):
    """
    用于对数据线性插值的函数
    """
    n_row = len(data.index)
    n_col = len(data.columns)
    res = np.zeros(shape=(n_row, n_col))

    for i in range(n_col):
        res[:, i] = data.iloc[:, i].values  # 提取列数据
        y = data.iloc[start:end, i]  # 选取要插值的范围
        location = np.where(y.notnull())[0]  # 获取y中非空数据的索引数组
        upper_bound = max(location)
        lower_bound = min(location)
        f = interp1d(location, y[y.notnull()], kind=method)  # 创建插值函数
        x = np.linspace(lower_bound, upper_bound, num=upper_bound - lower_bound, endpoint=False)
        res[lower_bound:upper_bound, i] = f(x)

    res = pd.DataFrame(res, index=data.index, columns=data.columns)
    return res


def standardize_data(df):
    """
    标准化数据，返回标准化后的DataFrame和统计量字典
    """
    stats = {
        'mean': df.mean(axis=0),
        'std': df.std(axis=0),
    }
    standardized_df = (df - stats['mean']) / stats['std']

    return standardized_df, stats



def rand_Matrix(n_row, n_col):
    """
    生成一个随机矩阵,用于初始化误差
    """
    randArr = np.random.randn(n_row, n_col)
    randMat = np.asmatrix(randArr)
    return randMat


def calculate_factor_loadings(observables, factors):
    """
    计算因子载荷的函数
    传入参数为数据和因子
    """
    n_time = len(observables.index)
    x = np.asmatrix(observables - observables.mean())
    F = np.asmatrix(factors)
    temp = F[0].T @ (F[0])
    for i in range(1, n_time):
        temp = temp + F[i].T @ (F[i])

    Lambda = x[0].T @ (F[0]) @ (temp.I)
    for i in range(1, n_time):
        Lambda = Lambda + x[i].T @ (F[i]) @ (temp.I)

    return Lambda


def calculate_prediction_matrix(factors):
    """
    用于计算隐含因子的状态转移方程的状态转移矩阵A
    """
    n_time = len(factors.index)
    F = np.asmatrix(factors)

    temp = F[0].T @ (F[0])
    for i in range(2, n_time):
        temp = temp + F[i - 1].T @ (F[i - 1])

    A = F[1].T @ (F[0]) @ (temp.I)
    for i in range(2, n_time):
        A = A + F[i].T @ (F[i - 1]) @ (temp.I)

    return A


def calculate_shock_matrix(factors, prediction_matrix, n_shocks):
    """
    用于计算隐含因子的状态转移方程的冲击矩阵B
    """
    n_time = len(factors.index)
    F = np.asmatrix(factors)
    A = prediction_matrix

    temp = F[0].T @ (F[0])
    for i in range(2, n_time):
        temp = temp + F[i - 1].T @ (F[i - 1])

    term1 = F[1].T @ (F[1])
    for i in range(2, n_time):
        term1 = term1 + F[i].T @ (F[i])
    term1 = term1 / (n_time - 1)
    term2 = A @ (temp / (n_time - 1)) @ (A.T)
    Sigma = term1 - term2  # 状态变量的协方差矩阵
    Sigma = (Sigma + Sigma.T) / 2  # 确保Sigma对称

    eigenvalues, eigenvectors = np.linalg.eigh(Sigma)  # 对sigma特征值分解
    # 仅保留非负特征值
    eigenvalues = np.maximum(eigenvalues, 0)
    sorted_indices = np.argsort(eigenvalues)
    evalues = eigenvalues[sorted_indices[:-n_shocks - 1:-1]]
    M = eigenvectors[:, sorted_indices[:-n_shocks - 1:-1]]
    B = M @ (np.diag(pow(evalues, 0.5)))  # 特征向量和对应的特征值平方根构成冲击矩阵

    return B, Sigma


def calculate_covariance(factors):
    """
    计算隐含因子的协方差矩阵
    """
    temp = [factors.iloc[:,i] for i in range(len(factors.columns))]
    return np.cov(temp)


def calculate_pca(observables, n_factors):
    """
    主成分分析
    用于获取初始的隐含因子
    """
    n_time = len(observables.index)

    z = np.asmatrix((observables - observables.mean()) / observables.std())

    S = z[0].T @ (z[0])
    for i in range(1, n_time):
        S = S + z[i].T @ (z[i])  # S为协方差矩阵

    eigenvalues, eigenvectors = np.linalg.eig(S)
    sorted_indices = np.argsort(eigenvalues)
    evalues = eigenvalues[sorted_indices[:-n_factors - 1:-1]]
    V = np.asmatrix(eigenvectors[:, sorted_indices[:-n_factors - 1:-1]])  # V为特征向量矩阵
    D = np.diag(evalues)  # D为对角矩阵，表示降维后数据在每个主成分上的方差大小

    return D, V, S


def process_data(data, monthly_column=None, w_d_column=None, start_date='2014-01-01', end_date='2024-12-31'):
    """
    处理数据的函数
    输入是 原始数据， 月度指标名称， 周/日度指标名称， 开始日期， 截止日期
    输出是 月度频率的 所有数据同比值的 一阶差分
    'start_date'取决于指标有效期
    """

    data_w_d = data[w_d_column].copy()
    data_m = data[monthly_column].copy()

    # 生成周频时间索引,每个月的日期范围并划分为4个“周”
    month_ranges = pd.date_range(start=start_date, end=end_date, freq='ME')
    weekly_dates = []

    for month_end in month_ranges:
        month_start = month_end.replace(day=1)
        all_days = pd.date_range(start=month_start, end=month_end)
        split_points = np.array_split(all_days, 4)  # 将一个月分为4个“周”
        for week in split_points:
            weekly_dates.append(week[-1])

    dates = pd.DatetimeIndex(weekly_dates)

    # 开始处理数据

    # 日度数据取平均转为周度
    data_weekly_daily = pd.DataFrame(data=np.nan, index=dates, columns=data_w_d.columns)
    for i in range(1, len(data_weekly_daily.index)):
        temp = data_w_d[(data_w_d.index.date > data_weekly_daily.index[i-1].date()) &
                        (data_w_d.index.date <= data_weekly_daily.index[i].date())
                        ].replace(0, np.nan).mean(skipna=True)
        data_weekly_daily.iloc[i] = temp.values

    # 月度数据转为周度
    data_monthly = pd.DataFrame(data=np.nan, index=dates, columns=data_m.columns)
    for i in range(1, len(data_monthly.index)):
        sub = data_m[(data_m.index.date > data_monthly.index[i-1].date()) &
                     (data_m.index.date <= data_monthly.index[i].date())
                     ].replace(0, np.nan).mean(skipna=True)
        data_monthly.iloc[i] = sub.values
    # 把月度数据中所有被记为nan值的0值还原为0
    data_monthly = data_monthly.apply(lambda row: row.fillna(0) if row.notna().any() else row, axis=1)

    # ----------------------------开始计算同比并取一阶差分----------------------------
    # 周度价格数据先转为同比
    temp_w_d = transform_data_yoy(data=data_weekly_daily)
    # 再计算相对上个月一阶差分
    temp_w_d_diff = temp_w_d.diff(periods=4)

    # 月度数据以同比形式下载，直接转换为差分
    temp_m = data_monthly.dropna(how='all')
    temp_m = temp_m.diff()
    # 将差分结果重新对齐到 data_monthly 的索引
    data_monthly.update(temp_m)

    # 合并周度月度数据
    processed_data_diff = data_monthly.join(temp_w_d_diff, how='outer')

    # 重采样至月频
    data_diff_monthly = processed_data_diff.resample('ME').mean()  # 默认skipna
    data_diff_monthly = data_diff_monthly.dropna(subset=data_w_d.columns[1:], how='all')

    return data_diff_monthly



def transform_data_yoy(data):
    """
    把数据从价格转换为同比的函数
    用于周度和日度数据
    """
    transform = pd.DataFrame(data=np.nan, index=data.index, columns=data.columns)
    for i in range(54, len(data.index)):
        last_year = data[data.index <= (data.index[i] - timedelta(360))].iloc[-1]  # 取出至少360天前的数据
        transform.iloc[i] = (data.iloc[i] - last_year) / last_year * 100

    return transform


def count_monthly_indicators(data_monthly):
    # 记录建模变量中不及时发布的月度指标的个数
    monthly_indicators = ['import', 'CPI', 'CPI_food', 'CPI_housing', 'CPI_transcomm', 'CPI_culture', 'housing_price',
                          'PPI','PPI_mop1','PPI_mop2','PPI_mop3','PPI_life','CCPI','export'
                          ]
    count_monthly = 0
    for column in data_monthly.columns:
        if column in monthly_indicators:
            count_monthly = count_monthly + 1

    return count_monthly



