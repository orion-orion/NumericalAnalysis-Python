'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-05-29 21:38:09
LastEditors: ZhangHongYu
LastEditTime: 2021-05-29 21:42:10
'''
import numpy as np
# 置换矩阵P乘法顺序
P1 = np.array(
    [
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0]
    ]
)
P2 = np.array(
    [
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0]
    ]
)

# print(np.matmul(A, B))
print(np.matmul(P2, P1)) #对
print(np.matmul(P1, P2)) #错
print(np.eye(3))