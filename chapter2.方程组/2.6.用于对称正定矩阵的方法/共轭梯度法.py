'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-06-05 18:56:58
LastEditors: ZhangHongYu
LastEditTime: 2021-06-06 16:31:46
'''
import numpy as np
from copy import deepcopy
A = np.array(
    [
        [2, 2],
        [2, 5],
    ]
)
assert(A.shape[0] == A.shape[1])
n = A.shape[0]
b = np.array([6, 3])
x = [0, 0] #初始估计解向量
d = r = b - A.dot(x)
for k in range(n):
    # 余项为0，可以返回
    if r.any() == 0 :
        break
    A_d = np.matmul(A, d) #先把Ad先计算出来
    # 计算更新步长参数alpha
    alpha = r.dot(r) / np.matmul(d, A_d)
    # 更新解向量
    x = x + alpha * d
    pre_r = np.copy(r)
    # 更新余项
    r = r - alpha * A_d
    # 计算更新步长参数belta
    belta = r.dot(r)/pre_r.dot(pre_r)
    # 更新d
    d = r + belta * d
print(x)
    
