import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from DynamicFactorModel import *
from backtest_functions import *


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.use('TkAgg')

"""
模拟每个指标逐步发布,每日更新当月CPI同比预测值
"""

file = 'data/CPI_data.xlsx'
data_all = pd.read_excel(file)
data_all.columns = ['date','import','CPI','CPI_food','CPI_housing','CPI_transcomm','CPI_culture','housing_price','PMI_business','CPI_expectation',
                    'hainan','gas93#','gas0#','brent','commodity','pork','beef','vegetables','fruits','egg']

# 记录月度数据更新时间
count_monthly = count_monthly_indicators(data_all)
# 第5行是数据更新时间
update_schedule = data_all.iloc[5, 1:count_monthly+1].astype(str).apply(lambda x: int(x.split('-')[2])).to_dict()

data_all = data_all[6:]
data_all['date'] = pd.to_datetime(data_all['date'])
data_all.set_index('date', inplace=True)

cpi_hist = data_all[['CPI', 'CPI_expectation']]
cpi_hist = cpi_hist.dropna(how='all').resample('ME').last()

# 月度数据和日度周度数据分开
weekly_daily_columns = ['hainan','gas93#','gas0#','brent','commodity','pork','beef','vegetables','fruits','egg']
monthly_columns = ['import','CPI','CPI_food','CPI_housing','CPI_transcomm','CPI_culture','housing_price','PMI_business','CPI_expectation']
count_monthly = count_monthly_indicators(data_all[monthly_columns])


def update_monthly_indicators_with_dates(temp_mon, current_date, update_schedule):
    """
    根据每个指标的具体发布日期模拟逐步发布
    update_schedule是包含每个指标具体发布日期的字典
    """
    previous_month_end = (current_date - pd.DateOffset(months=1)).replace(day=1) + pd.offsets.MonthEnd(0)

    # 遍历每个月度指标
    for col in temp_mon.columns:
        # 获取当前指标的滞后天数
        lag_days = update_schedule.get(col, 0)  # 默认滞后为0天

        # 计算前一个月数据的发布日期
        release_date = previous_month_end + pd.Timedelta(days=lag_days)

        # 如果当前日期早于发布日期，前一个月数据视为未发布（NaN）
        if current_date < release_date:
            temp_mon.loc[previous_month_end, col] = 0

    return temp_mon


def generate_dates(end_date):
    """
    生成回测相关日期参数
    """
    backtest_month = pd.Timestamp(end_date)  # 回测月的最后一天
    first_day = backtest_month.replace(day=1)  # 当月第一天
    last_month = first_day - timedelta(days=1)  # 上个月最后一天
    backtest_range = pd.date_range(start=first_day, end=backtest_month, freq='D')  # 回测期间

    return backtest_month, first_day, last_month, backtest_range


# ------------------------------ 回测 ------------------------------
# 创建一个储蓄回测结果的dataframe
backtest_month, first_day, last_month, backtest_range = generate_dates(end_date='2024-09-30')

# 初始化回测结果
backtest_results = pd.DataFrame(data=np.nan, index=backtest_range, columns=['真实值_差分', '预测值_差分', '真实值', '预测值'])


for dates in tqdm(backtest_range):
    print(f"start:{dates}")

    # 先记录该月CPI实现值
    backtest_results.loc[dates, '真实值_差分'] = data_all.loc[backtest_month, 'CPI'] - data_all.loc[last_month, 'CPI']
    backtest_results.loc[dates, '真实值'] = data_all.loc[backtest_month, 'CPI']

    # 截取回测时间点的历史数据
    temp_day = data_all[weekly_daily_columns].loc[data_all.index <= dates]
    temp_mon = data_all[monthly_columns].loc[data_all.index <= dates]
    temp_mon.loc[backtest_month, 'CPI_expectation'] = data_all.loc[backtest_month, 'CPI_expectation']  # 万德CPI一致预期可看作每天发布

    updated_mon = update_monthly_indicators_with_dates(temp_mon, dates, update_schedule)
    data_backtest = pd.concat([updated_mon, temp_day], axis=1)

    data_backtest = process_data(data_backtest, monthly_columns, weekly_daily_columns, end_date=backtest_month)

    backtest_cpi(data_backtest, cpi_hist, backtest_results, date=dates, types='CPI', n_mon=count_monthly)


# ------------------------------ 画出回测结果 ------------------------------
plot_compare(backtest_results.iloc[:,0], backtest_results.iloc[:, 1], f'CPI_差分_日度nowcasting_{backtest_month}',
             line_width=2.0, font_size='xx-large', dir='backtest_results')
print(backtest_results)

