'''
Descripttion: 多变量牛顿方法
Version: 1.0
Author: ZhangHongYu
Date: 2021-03-08 15:57:56
LastEditors: ZhangHongYu
LastEditTime: 2021-07-24 19:01:22
'''
import numpy as np
import math
import scipy
from scipy import linalg
# >>> matrix = np.array([[2, 3], [4, 5]])
# >>> a = np.array([2, 3])  (2, )可以右乘矩阵，自动做为(1, 2)对齐
# >>> a.dot(matrix)
# array([16, 21])
# 但是，如果是求外积一定要先reshape成矩阵形式
# 这个方法有一个好，就是不用求逆
# 注意这里x表x(k+1)，x0表x(k)，迭代小技巧可以使用
# 一定要注意x0和x分开存，且赋值不能x=x0，一定要用x = x0.copy()!!!
# 否则你这就变成失败的类SOR方法了，即错误的异步处理，算出来的值还一样?
def Broyden_2(x0, K, F): #迭代k次,包括x0在内共k+1个数
    # 初始向量
    x0 = x0.copy()
    x = x0.copy()
    # 初始矩阵，难以计算导数，用单位矩阵I初始化
    B = np.eye(x0.shape[0], dtype=np.float32)
    for k in range(K):
        x = x0 - np.matmul(B, F(x0))
        delta_f = F(x) - F(x0)
        delta_x = x - x0
        B = B + (delta_x - B.dot(delta_f)).reshape(-1, 1).dot(delta_x.reshape(1, -1)).dot(B)/delta_x.dot(B).dot(delta_f)
        x0 = x.copy()
    return x
def F(x:np.ndarray): 
    return np.array([x[1]-x[0]**3, x[0]**2+x[1]**2-1])
if __name__ == '__main__':
    # 初始解为[1, 1]结果才正确
    x0 = np.array([1, 1], dtype=np.float32)
    # 迭代次数
    K = 10 
    res = Broyden_2(x0, K, F)
    print(res)