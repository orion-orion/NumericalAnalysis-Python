'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-06-05 18:56:58
LastEditors: ZhangHongYu
LastEditTime: 2021-06-06 16:37:42
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
x = [0, 0] # 初始估计解向量
r = b - A.dot(x)
M = np.diag(np.diag(A)) # M 为A的对角矩阵
z = np.matmul(np.matrix(M).I.A, r)
d = np.copy(z)
for k in range(n):
    # 余项为0，可以返回
    if r.any() == 0 :
        break
    A_d = np.matmul(A, d)
    alpha = r.dot(z)/d.dot(A_d)
    x = x + alpha * d
    r_pre = np.copy(r)
    r = r - alpha * A_d
    z_pre = np.copy(z)
    #如果用matrix dot ndarray会返回二维matrix
    #如此对结果需用 .A取出ndarray并reshape到一维
    z = np.matmul(np.matrix(M).I.A, r) 
    belta = r.dot(z)/r_pre.dot(z_pre)
    d = z + belta*d
print(x)
    
