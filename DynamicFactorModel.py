from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
from datetime import timedelta
from statsmodels.tsa.api import VAR
import scipy
import sklearn
import statsmodels.api as sm
from tqdm import tqdm
from DiscreteKalmanFilter import *
from Functions import calculate_pca, calculate_factor_loadings, calculate_covariance


def DFM(observation, n_factors, n_shocks):
    """
    动态因子模型中状态转移方程的估计
    输入：数据，隐含因子个数，冲击个数
    输出：pca得到的初始隐含因子；因子载荷：状态转移方程的参数
    """
    # model: y_it = miu_i + lambda_i*F_t + e_it
    if len(observation.columns) <= n_factors:
        return print('Error: number of common factors exceeds limit')

    "pca"
    # 对原始周度数据标准化
    z = np.asmatrix((observation - observation.mean()) / observation.std())

    D, V, S = calculate_pca(observation, n_factors)
    Psi = np.asmatrix(np.diag(np.diag(S - V.dot(D).dot(V.T))))  # S 中无法通过主成分解释的剩余方差（也称为特异性方差）
    factors = V.T.dot(z.T).T
    CommonFactors = pd.DataFrame(data=factors, index=observation.index,
                                 columns=['Factor' + str(i + 1) for i in range(n_factors)])
    # 将标准化后的数据投影到主成分方向上，从而计算出每个观测点在各个主成分上的得分，得到common factor

    "Factors loadings"
    Lambda = calculate_factor_loadings(observation, CommonFactors)  # 计算因子载荷矩阵

    "VAR"
    # model: F_t = A*F_{t-1} + B*u_t
    # calculate matrix A and B
    # 状态转移方程符合VAR过程
    A = calculate_prediction_matrix(CommonFactors)
    B, Sigma = calculate_shock_matrix(CommonFactors, A, n_shocks)

    return DFMResultsWrapper(common_factors=CommonFactors, Lambda=Lambda, A=A, B=B, idiosyncratic_covariance=Psi,
                             prediction_covariance=Sigma, obs_mean=observation.mean())


class DFMResultsWrapper():
    def __init__(self, common_factors, Lambda, A, B, idiosyncratic_covariance, prediction_covariance, obs_mean):
        self.common_factors = common_factors
        self.Lambda = Lambda
        self.A = A
        self.B = B
        self.idiosyncratic_covariance = idiosyncratic_covariance
        self.prediction_covariance = prediction_covariance
        self.obs_mean = obs_mean


def DFM_EMalgo(observation, n_factors, n_shocks, n_iter, error='False'):
    """
    使用EM算法估计模型
    输出状态转移方程的各个参数和平滑后的隐含因子
    """
    # 计算初始的因子和因子载荷，以及估计状态转移方程
    dfm = DFM(observation, n_factors, n_shocks)

    if error == 'True':  # 初始的误差矩阵是随机数还是0，默认为0
        error = pd.DataFrame(data=rand_Matrix(len(observation.index), n_shocks),
                             columns=['shock' + str(i + 1) for i in range(n_shocks)], index=observation.index)
    else:
        error = pd.DataFrame(data=np.zeros(shape=(len(observation.index), n_shocks)),
                             columns=['shock' + str(i + 1) for i in range(n_shocks)], index=observation.index)

    # 卡尔曼滤波更新隐含因子估计
    kf = KalmanFilter(Z=observation - observation.mean(), U=error, A=dfm.A, B=dfm.B, H=dfm.Lambda,
                      state_names=dfm.common_factors.columns, x0=dfm.common_factors.iloc[0],
                      P0=calculate_covariance(dfm.common_factors), Q=dfm.prediction_covariance,
                      R=dfm.idiosyncratic_covariance)
    # 基于已知信息后向平滑因子估计
    fis = FIS(kf)

    # EM算法
    for i in range(n_iter):
        em = EMstep(fis, n_shocks)
        start = em.Lambda.I.dot((em.z - em.z.mean()).T).T
        kf = KalmanFilter(Z=observation - observation.mean(), U=error, A=em.A, B=em.B, H=em.Lambda,
                          state_names=dfm.common_factors.columns, x0=start[0], P0=calculate_covariance(em.x_sm), Q=em.Q,
                          R=em.R)
        fis = FIS(kf)

    return DFMEMResultsWrapper(A=em.A, B=em.B, Q=em.Q, R=em.R, Lambda=em.Lambda, x=kf.x, x_sm=em.x_sm, z=kf.z)


class DFMEMResultsWrapper():
    def __init__(self, A, B, Q, R, Lambda, x, x_sm, z):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.Lambda = Lambda
        self.x = x
        self.x_sm = x_sm
        self.z = z


def RevserseTranslate(Factors, miu, Lambda, names):
    """
    根据估计的隐含因子进行预测
    """
    observation = pd.DataFrame(data=Lambda.dot(Factors.T).T, columns=names, index=Factors.index)
    observation = observation + miu
    return observation

