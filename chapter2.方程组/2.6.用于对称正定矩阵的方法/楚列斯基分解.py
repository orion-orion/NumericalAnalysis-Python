'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-06-05 18:56:58
LastEditors: ZhangHongYu
LastEditTime: 2021-06-05 19:33:08
'''
import numpy as np
from copy import deepcopy
A = np.array(
    [
        [4, -2, 2],
        [-2, 2, -4],
        [2, -4, 11]
    ]
)
R = np.zeros(A.shape)
n = 3
for k in range(n):
    if A[k, k]<0:
        raise ValueError("对角线元素不能为负!")
        break
    R[k, k] = np.sqrt(A[k, k])
    u_T = deepcopy(A[k, k+1:n]/R[k, k]).reshape(1, -1)
    R[k, k+1:n] = u_T
    A[k+1:n, k+1:n] = A[k+1:n, k+1:n] - (u_T.T).dot(u_T)
print(R)

print(R.T.dot(R))
    
