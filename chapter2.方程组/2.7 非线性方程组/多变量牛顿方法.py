'''
Descripttion: 多变量牛顿方法
Version: 1.0
Author: ZhangHongYu
Date: 2021-03-08 15:57:56
LastEditors: ZhangHongYu
LastEditTime: 2021-10-17 19:54:23
'''
import numpy as np
import math
import torch
import scipy
from scipy import linalg
from torch.autograd.functional import jacobian
def multi_variable_newton(x0, K, F): #迭代k次,包括x0在内共k+1个数
    # 初始x张量
    x = torch.tensor(x0, requires_grad=True)
    for k in range(K):
        y = F(x)
        # 因为只能由标量函数backward，此处需要传入参数
        y.backward(torch.ones_like(x), retain_graph=True)
        # 计算Jocobian矩阵
        J = jacobian(F, x)
        with torch.no_grad(): 
            # 如果为了避免求逆，也可以解线性方程组Jv = -y，使x+v
            # 注意，此处y是一维，则返回的v也是一维
            v = np.linalg.solve(J, -y)
            x.add_(torch.tensor(v))
            # 这等价于
            # x.sub_(torch.matmul(torch.inverse(J), y))
        x.grad.zero_() 
    return x.detach().numpy()

def F(x): 
    # 书上定义的向量函数F(x): F(u, v)=(v-u**3, u**2+v**2-1)
    # 注意，这里不能重新定义torch.tensor对象，否则梯度无法传播
    # 故我们这里重新定义F(x): F(u, v)=(v-5u, 2u+2v)
    # 可以写成矩阵乘的形式，可以使梯度正常传播
    A = torch.tensor([[-5, 1], [2, 2]], dtype=torch.float32)
    return torch.matmul(A, x)
    
if __name__ == '__main__':
    # 初始解
    x0 = np.array([1, 2], dtype=np.float32)
    # 迭代次数
    K = 10 
    res = multi_variable_newton(x0, K, F)
    print(res)