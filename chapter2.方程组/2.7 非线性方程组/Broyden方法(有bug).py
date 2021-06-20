'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-06-19 19:22:08
LastEditors: ZhangHongYu
LastEditTime: 2021-06-19 20:42:52
'''
'''
Descripttion: 多变量牛顿方法
Version: 1.0
Author: ZhangHongYu
Date: 2021-03-08 15:57:56
LastEditors: ZhangHongYu
LastEditTime: 2021-06-19 14:52:43
'''
import numpy as np
import math
import torch
import scipy
from scipy import linalg
# 注意这里x表x(k+1)，x0表x(k)，迭代小技巧可以使用
# 一定要注意x0和x分开存，且赋值不能x=x0，一定要用x = x0.copy()!!!
def Broyden(x0, K, F): #迭代k次,包括x0在内共k+1个数
    # 初始向量
    x0 = x0.copy()
    x = x0.copy()
    # 初始矩阵
    A = np.random.rand(x0.shape[0], x0.shape[0])
    for k in range(K):
        print(np.matrix(A).I.A, F(x0))
        # 遇到A不可逆奈何?
        x = x0 - np.matmul(np.matrix(A).I.A, F(x0))
        delta_f = F(x) - F(x0)
        delta_x = x - x0
        A = A + (delta_f - A.dot(delta_x)).dot(delta_x.T)/delta_x.dot(delta_x)
        x0 = x.copy()
    return x
def F(x:np.ndarray): 
    return np.array([x[1]-x[0]**3, x[0]**2+x[1]**2-1])
if __name__ == '__main__':
    # 初始解
    x0 = np.array([1, 2], dtype=np.float32)
    # 迭代次数
    K = 10 
    res = Broyden(x0, K, F)
    print(res)