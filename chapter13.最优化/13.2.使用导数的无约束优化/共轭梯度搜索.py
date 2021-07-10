'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-07-02 22:03:03
LastEditors: ZhangHongYu
LastEditTime: 2021-07-03 15:22:14
'''
import numpy as np
import math
import torch

#x.grad为Dy/dx(假设Dy为最后一个节点)
def Conjugate_gradient_search(x0, k, f, alpha): #迭代k次,包括x0在内共k+1个数
    # 初始化计算图参数
    x = torch.tensor(x0, requires_grad=True)
    y = f(x)
    y.backward()
    r = -x.grad.detach().numpy()
    d = r.copy()
    for i in range(1, k+1):
        with torch.no_grad(): 
            x.add_(alpha*torch.tensor(d))
        y = f(x)
        y.backward() 
        old_r = r.copy()
        r = -x.grad.detach().numpy()  
        belta = r.dot(r)/old_r.dot(old_r)
        d = r + belta * d
    x_star = x.detach().numpy()
    minimum = f(x_star)
    # 最后得到的x即极值点的x
    return minimum, x_star

# 多元函数，但非向量函数
def f(x):
    return 5*x[0]**4 + 4*x[0]**2*x[1] - x[0]*x[1]**3 + 4*x[1]**4 - x[0]

if __name__ == '__main__':
    x0 = np.array([1.0, -1.0])
    k = 5 # k为迭代次数
    alpha = 0.01 # ita为迭代步长
    minimum, x_star = Conjugate_gradient_search(x0, k, f, alpha)
    print("the minimum is %.5f, the x_star is: ( %.5f, %.5f)"\
        % (minimum, x_star[0], x_star[1]))