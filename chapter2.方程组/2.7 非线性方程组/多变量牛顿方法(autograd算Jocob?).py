'''
Descripttion: 多变量牛顿方法
Version: 1.0
Author: ZhangHongYu
Date: 2021-03-08 15:57:56
LastEditors: ZhangHongYu
LastEditTime: 2021-06-06 22:27:08
'''
import numpy as np
import math
import torch
import scipy
# 如何用autograd算Jocobbi矩阵?
def multi_variable_newton(x0 ,t, f1, f2): #迭代t次,包括x0在内共t+1个数
    # 初始化计算图参数
    x = torch.tensor(x0, requires_grad=True)
    for i in range(1, t+1):
        y1, y2 = f1(x[0], x[1]), f2(x[0], x[1])
        
        y1.backward()
        y2.backward()
        
        s = 
        with torch.no_grad(): 
            x.add_(s)   
        x.grad.zero_()
    return x.detach().numpy()[0]
if __name__ == '__main__':
    f1 = lambda u, v: v - u**3
    f2 = lambda u, v: u**2 + v**2
    # 初始解
    x0 = np.array([1, 2])
    # 迭代次数
    t = 10 
    res = multi_variable_newton(x0, t, f1, f2)
    # print(res)