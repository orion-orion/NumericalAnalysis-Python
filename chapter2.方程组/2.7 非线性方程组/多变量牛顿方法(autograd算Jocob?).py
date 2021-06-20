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
def multi_variable_newton(x0, K, F): #迭代k次,包括x0在内共k+1个数
    # 初始x张量
    x = torch.tensor(x0, requires_grad=True)
    # F对x_tensor求导的Jocobi矩阵
    Jocobi_matrix = np.zeros((x.shape[0], x.shape[0]))
    b = np.zeros((x.shape[0], 1))
    for k in range(K):
        y = F(x)
        print(y)
        # 因为只有标量能定义backward
        dummy = y.sum()
        dummy.backward()
        # 更新Jocobi矩阵
        with torch.no_grad(): 
            print(x.grad)
            print(x)
            Jocobi_matrix[0, :] = x.grad.numpy()[0, :]
            Jocobi_matrix[1, :] = x.grad.numpy()[1, :]
            b = y.numpy().reshape(-1, 1)
            # 解线性方程组DF(x(k)) * s = -F(x(k))
            s = np.linalg.solve(Jocobi_matrix, -b)
            # x(k+1) = x(k) + s
            x.add_(x, torch.tensor(s.reshape(-1,)))
        x.grad.zero_() 
    return x
def F(x:torch.tensor): 
    # 书上定义的向量函数F(x): F(u, v)=(v-u**3, u**2+v**2-1)
    # 该向量函数不能表示为矩阵乘，重新定义tensor梯度传播会出现问题
    # F(u, v) = (u**2, v**2)这种都不行，因为求导得到向量而不是矩阵
    # 故这里为 F(u, v) = (2u + 2v, 3u + 2v)
    # A=[[2, 2], [2, 3]]
    A = torch.tensor([[2, 2], [2, 3]], dtype=torch.float32)
    # torch自动处理x维度问题
    return torch.matmul(A, x)
if __name__ == '__main__':
    # 初始解
    x0 = np.array([1, 2], dtype=np.float32)
    # 迭代次数
    K = 10 
    res = multi_variable_newton(x0, K, F)
    print(res)