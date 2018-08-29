# -*- coding:utf-8 -*-


'''
Created on Feb 20, 2017
最小二乘拟合 给定的函数 fit_fun(x)
已知数据X（2xN）,Y(1xN),可直接计算直线的参数a和b
直接计算的公式：ab = inv(XXt)*X*Yt
@author: Xiankai Chen
@email: xiankai.chen@qq.com
'''
import numpy as np
import matplotlib.pyplot as plt


def fun2ploy(x, n):
    '''
    数据转化为[x^0,x^1,x^2,...x^n]
    '''
    lens = len(x)
    X = np.ones([1, lens]);
    for i in range(1, n):
        X = np.vstack((X, np.power(x, i)))
    return X


def leastseq_byploy(x, y, ploy_dim):
    '''
    最小二乘求解
    '''
    # 散点图
    plt.scatter(x, y, color="r", marker='o', s=50)

    X = fun2ploy(x, ploy_dim);
    # 直接求解
    Xt = X.transpose();
    XXt = X.dot(Xt);
    XXtInv = np.linalg.inv(XXt)
    XXtInvX = XXtInv.dot(X)
    coef = XXtInvX.dot(y.T)

    y_est = Xt.dot(coef)

    return y_est, coef


def fit_fun(x):
    '''
    要拟合的函数
    '''
    # return np.power(x,5)
    return np.sin(x)
    # return 5*x+3


if __name__ == '__main__':
    data_num = 100;
    ploy_dim = 10;
    noise_scale = 0.2;
    ## 数据准备
    x = np.array(np.linspace(-2 * np.pi, 2 * np.pi, data_num))
    y = fit_fun(x) + noise_scale * np.random.rand(1, data_num)
    # 最小二乘拟合
    [y_est, coef] = leastseq_byploy(x, y, 10)

    # 显示拟合结果
    org_data = plt.scatter(x, y, color="r", marker='o', s=50)
    est_data = plt.plot(x, y_est, color="b", linewidth=3)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Fit funtion with leastseq method")
    plt.legend(["Noise data", "Fited function"]);
    plt.show()