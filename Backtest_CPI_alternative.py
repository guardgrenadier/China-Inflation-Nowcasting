import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from CPI_data import CPI_data_diff, count_monthly, cpi_hist
from DiscreteKalmanFilter import *
from Functions import *
from DynamicFactorModel import *


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.use('TkAgg')


def DFM_EM_alt(data, cpi_history, backtest_res, date=None, types=None, n_mon=None,
               n_factors=6, n_shocks=2, n_iter=1000, tol=1e-8):
    """
    另一种模型收敛方式：
    E步：固定其他参数，通过卡尔曼滤波和后向平滑更新隐含因子估计
    M步：固定隐含因子，更新动态因子模型参数
    执行EM，直到隐含因子收敛（两次隐含因子估计值的差小于参数tol）
    """
    # 对数据进行线性插值填充nan值
    data_filled = DataInterpolation(data, 0, len(data.index), 'linear').dropna(axis=0, how='any')

    # 第一次估计动态因子模型
    dfm_results = DFM(data_filled, n_factors, n_shocks)

    # 初始化pre_factors
    prev_factors = dfm_results.common_factors.copy()
    empty_row = pd.DataFrame([[np.nan] * prev_factors.shape[1]], columns=prev_factors.columns)
    prev_factors = pd.concat([prev_factors, empty_row], ignore_index=True)

    # ------------------------------ EM算法 ------------------------------
    for iteration in range(n_iter):
        kf = KalmanFilter(
            Z=data_filled - data_filled.mean(),
            U=pd.DataFrame(0, index=data.index, columns=[f'shock{i + 1}' for i in range(n_shocks)]),
            A=dfm_results.A,
            B=dfm_results.B,
            H=dfm_results.Lambda,
            state_names=dfm_results.common_factors.columns,
            x0=dfm_results.common_factors.iloc[0],
            P0=calculate_covariance(dfm_results.common_factors),
            Q=dfm_results.prediction_covariance,
            R=dfm_results.idiosyncratic_covariance
        )

        fis = FIS(kf)

        em = EMstep(fis, n_shocks)

        predict = RevserseTranslate(kf.x, data_filled.mean(), em.Lambda, data.columns)

        data.iloc[-1, :n_mon] = predict.iloc[-1, :n_mon]

        data_filled = DataInterpolation(data, 0, len(data.index), 'linear').dropna(axis=0, how='any')

        dfm_results = DFM(data_filled, n_factors, n_shocks)

        # 检查隐含因子是否收敛
        factor_diff = np.linalg.norm(dfm_results.common_factors.values - prev_factors.values)
        if factor_diff < tol:
            break

        prev_factors = dfm_results.common_factors.copy()

    previous_date = pd.Timestamp(date) - pd.offsets.MonthEnd(1)

    # 保存Nowcasting的值
    predicted_diff = data.iloc[-1, 1]
    backtest_res.loc[date, '预测值_差分'] = predicted_diff
    backtest_res.loc[date, '预测值'] = cpi_history.loc[previous_date, types] + predicted_diff


# ------------------------------ 回测 ------------------------------
if __name__ == '__main__':
    # 创建一个储蓄回测结果的dataframe
    backtest_range = pd.date_range(start="2023-11-01", end="2024-12-31", freq='ME')  # 回测范围
    backtest_results = pd.DataFrame(data=np.nan, index=backtest_range, columns=['真实值_差分', '预测值_差分', '真实值', '预测值'])

    for dates in tqdm(backtest_range):
        print(f"start:{dates}")
        # 截取回测时间点的历史数据
        data_backtest = CPI_data_diff.copy()[CPI_data_diff.index <= dates]

        # 先记录该月CPI实现值
        backtest_results.loc[dates, '真实值_差分'] = data_backtest.iloc[-1, 1]
        backtest_results.loc[dates, '真实值'] = cpi_hist.loc[dates, 'CPI']

        # 该月月度数据数据未发布（除PMI和预期，回测时间点为t+1月第一天）
        data_backtest.iloc[-1, :count_monthly] = np.nan

        # 对当月CPI进行nowcast并存下nowcast结果，选择隐含因子个数
        DFM_EM_alt(data_backtest, n_factors=5, date=dates)
        previous_dates = dates

    # ------------------------------ 画出回测结果 ------------------------------
    plot_compare(backtest_results.iloc[:, 0], backtest_results.iloc[:, 1], 'CPI_差分_alt',
                 line_width=2.0,font_size='xx-large', dir='backtest_results')
    plot_compare(backtest_results.iloc[:, 2], backtest_results.iloc[:, 3], 'CPI同比_alt',
                 line_width=2.0,font_size='xx-large')
    print(backtest_results)

    # 保存结果
    os.makedirs('backtest_results', exist_ok=True)
    #backtest_results.to_excel('backtest_results/CPI_backtest_alt_results.xlsx')

