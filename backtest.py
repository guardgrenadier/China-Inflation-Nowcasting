import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from Backtest_CPI_alternative import DFM_EM_alt
from backtest_functions import *
from DiscreteKalmanFilter import *
from Functions import *
from DynamicFactorModel import *

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.use('TkAgg')


def run_backtest(data_diff, hist_data, backtest_range, count_monthly, indicator_type, n_factors, alt_method=False):
    """
    回测函数
    data_diff: 差分后的数据
    hist_data: cpi/ppi真实历史数据（实现值）
    n_factors: DFM隐含因子个数
    indicator_type: 'CPI' 或 'PPI'
    backtest_range: 回测日期范围
    count_monthly: 月度指标数量
    alt_method: 是否使用另一种模型收敛方法
    """
    backtest_results = pd.DataFrame(data=np.nan, index=backtest_range,
                                    columns=['真实值_差分', '预测值_差分', '真实值', '预测值'])

    for dates in tqdm(backtest_range, desc=f'{indicator_type}回测中'):

        # 截取回测时间点的历史数据
        data_backtest = data_diff.copy()[data_diff.index <= dates]

        # 先记录该月真实值
        backtest_results.loc[dates, '真实值_差分'] = data_backtest.iloc[-1, 1]
        backtest_results.loc[dates, '真实值'] = hist_data.loc[dates, indicator_type]

        # 该月月度数据数据未发布（除PMI和预期，回测时间点为t+1月第一天）
        data_backtest.iloc[-1, :count_monthly] = np.nan

        if alt_method:
            # 使用替代方法
            DFM_EM_alt(data_backtest, hist_data, backtest_results, n_factors=n_factors,
                       date=dates, types=indicator_type, n_mon=count_monthly)
        else:
            # 对当月指标进行nowcasting并存下nowcasting结果
            backtest_cpi(data_backtest, hist_data, backtest_results, n_factors=n_factors,
                         date=dates, types=indicator_type, n_mon=count_monthly)

    return backtest_results


def plot_backtest_results(backtest_results, indicator_type, alt_method=False):
    """
    绘制回测结果
    """
    suffix = '_alt' if alt_method else ''

    # 绘制差分结果
    plot_compare(backtest_results.iloc[:, 0], backtest_results.iloc[:, 1],
                 f'{indicator_type}_差分{suffix}', line_width=2.0, font_size='xx-large', dir='backtest_results')

    # 绘制同比结果
    plot_compare(backtest_results.iloc[:, 2], backtest_results.iloc[:, 3],
                 f'{indicator_type}同比{suffix}', line_width=2.0, font_size='xx-large', dir='backtest_results')

    # 保存回测结果
    os.makedirs('backtest_results', exist_ok=True)
    backtest_results.to_excel(f'backtest_results/{indicator_type}_backtest_results{suffix}.xlsx')
