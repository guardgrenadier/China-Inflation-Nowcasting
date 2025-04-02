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
from Functions import *


class KalmanFilterResultsWrapper():
    def __init__(self, x_minus, x, z, Kalman_gain, P, P_minus, state_names, A):
        self.x_minus = x_minus
        self.x = x
        self.z = z
        self.Kalman_gain = Kalman_gain
        self.P = P
        self.state_names = state_names
        self.A = A
        self.P_minus = P_minus


    """
    卡尔曼滤波算法
    输入：状态转移方程各参数：隐含因子第一列的值
    """
def KalmanFilter(Z, U, A, B, H, state_names, x0, P0, Q, R):\
    # x_t = A*x_{t-1} + B*u_t + Q
    # z_t = H*x_t + R
    # Q is process noise covariance
    # R is measurement noice covariance

    n_time = len(Z.index)
    n_state = len(state_names)
    z = np.asmatrix(Z.values)
    u = np.asmatrix(U.values)

    # predictions
    x = np.asmatrix(np.zeros(shape=(n_time, n_state)))
    x[0]=x0
    x_minus = np.asmatrix(np.zeros(shape=(n_time, n_state)))
    x_minus[0] = x0

    # factor errors
    P = [0 for i in range(n_time)]
    P[0] = P0
    P_minus = [0 for i in range(n_time)]
    P_minus[0] = P0

    # Kalman gains
    K = [0 for i in range(n_time)]

    # "Kalman Filter"
    for i in range(1,n_time):
        # check if z is available
        ix = np.where(1-np.isnan(z[i]))[1]
        z_t = z[i,ix]
        H_t = H[ix]
        R_t = R[ix][:,ix]

        # "prediction step"
        x_minus[i] = (A@(x[i-1].T)).T + (B@(u[i].T)).T
        P_minus[i] = A@(P[i-1])@(A.T) + Q

        # "update step"
        temp = H_t@(P_minus[i])@(H_t.T) + R_t
        K[i] = P_minus[i]@(H_t.T)@(temp.I)
        P[i] = P_minus[i] - K[i]@(H_t)@(P_minus[i])
        x[i] = (x_minus[i].T + K[i]@(z_t.T-H_t@(x_minus[i].T))).T

    x = pd.DataFrame(data=x, index=Z.index, columns=state_names)
    x_minus = pd.DataFrame(data=x_minus, index=Z.index, columns=state_names)

    return KalmanFilterResultsWrapper(x_minus=x_minus, x=x, z=Z, Kalman_gain=K, P=P, P_minus=P_minus, state_names=state_names, A=A)


def FIS(res_KF):
    """
    平滑卡尔曼滤波后的隐含因子
    输入为卡尔曼滤波的结果
    输出为平滑后的因子
    """
    N = len(res_KF.x.index)
    n_state = len(res_KF.x.columns)
    x = np.asmatrix(res_KF.x)
    x_minus = np.asmatrix(res_KF.x_minus)
    P = res_KF.P
    P_minus = res_KF.P_minus


    x_sm = np.asmatrix(np.zeros(shape=(N, n_state)))
    x_sm[N-1] = x[N-1]

    P_sm = [0 for i in range(N)]
    P_sm[N-1] = P[N-1]

    J = [0 for i in range(N)]

    for i in reversed(range(N-1)):
        J[i] = P[i]@(res_KF.A.T)@(P_minus[i+1].I)
        P_sm[i] = P[i] - J[i]@(P_minus[i+1]-P_sm[i+1])@(J[i].T)
        x_sm[i] = (x[i].T + J[i]@(x_sm[i+1].T-x_minus[i+1].T)).T

    x_sm = pd.DataFrame(data=x_sm, index=res_KF.x.index, columns=res_KF.x.columns)

    return SKFResultsWrapper(x_sm=x_sm, P_sm=P_sm,z=res_KF.z)

class SKFResultsWrapper():
    def __init__(self, x_sm, P_sm, z):
        self.x_sm = x_sm
        self.P_sm = P_sm
        self.z = z



def EMstep(res_SKF, n_shocks):
    """
    执行EM算法的函数
    输入：FIS后的结果；冲击个数
    输出：基于更新的因子重新估计的状态转移矩阵和DFM的参数
    """
    f = res_SKF.x_sm
    y = res_SKF.z

    Lambda = calculate_factor_loadings(y, f)
    A = calculate_prediction_matrix(f)
    B, Q = calculate_shock_matrix(f, A, n_shocks)

    resid = (np.asmatrix(y).T - Lambda.dot(np.asmatrix(f).T)).T
    R = np.diag(np.diag(np.cov(resid.T)))

    return EMstepResultsWrapper(Lambda=Lambda, A=A, B=B, Q=Q, R=R, x_sm=f, z=y)

class EMstepResultsWrapper():
    def __init__(self, Lambda, A, B, Q, R, x_sm, z):
        self.Lambda = Lambda
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.x_sm = x_sm
        self.z = z

