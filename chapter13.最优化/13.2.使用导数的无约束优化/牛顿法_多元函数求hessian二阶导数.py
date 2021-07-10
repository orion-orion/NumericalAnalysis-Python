'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-07-02 22:03:03
LastEditors: ZhangHongYu
LastEditTime: 2021-07-10 11:22:48
'''
import numpy as np
import math
import torch
# from torch.autograd.functional import hessian
from torch.autograd import grad

def nth_derivative(func, wrt, n):
    for i in range(n):
        grads = grad(func, wrt, create_graph=True)[0]
        func = grads.sum()
    return grads

# 多元函数，但非向量函数
def f(x):
    return 5*x[0]**4 + 4*x[0]**2*x[1] - x[0]*x[1]**3 + 4*x[1]**4 - x[0] 

#x.grad为Dy/dx(假设Dy为最后一个节点)
def gradient_descent(x0, k, f, eta): #迭代k次,包括x0在内共k+1个数
    # 初始化计算图参数
    x = torch.tensor(x0, requires_grad=True)
    for i in range(1, k+1):
        y = f(x)
        # y.backward(retain_graph=True) 
        with torch.no_grad(): 
            # 1阶导数可以直接访问x.grad
            # 高阶倒数我们需要调用nth_derivative接口，这里返回hession矩阵
            # 注意，Hession矩阵要求逆
            H = nth_derivative(y, x, 2)
            print(H)
            v = np.linalg.solve(H, -x.grad)
            x.add_(v)
            # x.sub_(eta*torch.matmul(
            #     x.grad.reshape(1, -1), torch.inverse(hessian(f, x))))
        x.grad.zero_() 
        x.grad.grad.zero_()
    x_star = x.detach().numpy()
    return f(x_star), x_star 

if __name__ == '__main__':
    x0 = np.array([1.0, 1.0])
    k = 25 # k为迭代次数
    eta = 0.01 # ita为迭代步长
    minimum, x_star = gradient_descent(x0, k, f, eta)
    print("the minimum is %.5f, the x_star is: ( %.5f, %.5f)"\
        % (minimum, x_star[0], x_star[1]))