'''
Descripttion: 最速下降法，此处用到求多元函数的梯度
Version: 1.0
Author: ZhangHongYu
Date: 2021-07-02 22:03:03
LastEditors: ZhangHongYu
LastEditTime: 2021-07-24 19:12:38
'''
import numpy as np
import math
import torch

#x.grad为Dy/dx(假设Dy为最后一个节点)
def gradient_descent(x0, k, f, eta): #迭代k次,包括x0在内共k+1个数
    # 初始化计算图参数
    x = torch.tensor(x0, requires_grad=True)
    for i in range(1, k+1):
        y = f(x)
        y.backward() 
        with torch.no_grad(): 
            x.sub_(eta*x.grad)
        x.grad.zero_()  #这里的梯度必须要清0，否则计算是错的
    x_star = x.detach().numpy()
    return f(x_star), x_star 

# 多元函数，但非向量函数
def f(x):
    return 5*x[0]**4 + 4*x[0]**2*x[1] - x[0]*x[1]**3 + 4*x[1]**4 - x[0]

if __name__ == '__main__':
    x0 = np.array([1.0, -1.0])
    k = 25 # k为迭代次数
    eta = 0.01 # ita为迭代步长
    minimum, x_star = gradient_descent(x0, k, f, eta)
    print("the minimum is %.5f, the x_star is: ( %.5f, %.5f)"\
        % (minimum, x_star[0], x_star[1]))