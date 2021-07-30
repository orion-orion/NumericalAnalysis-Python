'''
Descripttion: Broyden方法
Version: 1.0
Author: ZhangHongYu
Date: 2021-06-19 19:22:08
LastEditors: ZhangHongYu
LastEditTime: 2021-07-24 19:36:09
'''
import numpy as np
import math
import torch
import scipy
from scipy import linalg
# 注意这里x表x(k+1)，x0表x(k)，迭代小技巧可以使用
# 一定要注意x0和x分开存，且若是直接赋值不能用x=x0，一定要用x = x0.copy()!!!
# 最后还要注意，如果是向量求外积一定要先reshape成矩阵形式
def Broyden(x0, K, F): #迭代k次,包括x0在内共k+1个数
    # 初始向量
    x0 = x0.copy()
    x = x0.copy()
    # 初始矩阵，我们将其初始化为对角阵(保证可逆)
    A = np.eye(x0.shape[0])
    for k in range(K):
        # 这里A做为Jocobian矩阵的估计
        # 此处A必须可逆，不可逆那么算法就会出问题
        if np.linalg.det(A)==0:
            raise RuntimeError("A is a singular matrix!")
        x = x0 - np.matmul(np.matrix(A).I.A, F(x0))
        delta_f = F(x) - F(x0)
        delta_x = x - x0
        A = A + (delta_f - A.dot(delta_x)).reshape(-1, 1).dot(delta_x.reshape(1, -1))/delta_x.dot(delta_x)
        x0 = x.copy()
    return x
def F(x:np.ndarray): 
    return np.array([x[1]-x[0]**3, x[0]**2+x[1]**2-1])
if __name__ == '__main__':
    # 初始解，初始值要设成(1, 1)才能结果正确
    x0 = np.array([1, 1], dtype=np.float32)
    # 迭代次数
    K = 10 
    res = Broyden(x0, K, F)
    print(res)