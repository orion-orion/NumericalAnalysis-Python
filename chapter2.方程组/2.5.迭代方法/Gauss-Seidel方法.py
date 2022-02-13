'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-03-08 17:26:11
LastEditors: ZhangHongYu
LastEditTime: 2021-05-30 16:31:29
'''
import numpy as np
import copy
eps = 1e-6
n = 6

# 向量各元素之间互相依赖，不可并行/向量化
def GaussSeidel(A, b):
    assert(A.shape[0] == A.shape[1] == b.shape[0])
    x = np.zeros(b.shape, dtype=np.float32)
    # 迭代次数
    for t in range(n):
        for i in range(x.shape[0]):
            x[i] = b[i]
            for j in range(A.shape[1]):
                if j != i :
                    x[i] -= A[i, j] * x[j]
            x[i] /= A[i][i]
    return x    

if __name__ == '__main__':
    # A一定要是主对角线占优矩阵
    A = np.array(
        [
            [3, 1, -1],
            [2, 4, 1],
            [-1, 2, 5]
        ],dtype=np.float32
    )
    b = np.array(
        [4, 1, 1],dtype=np.float32
    )
    x = GaussSeidel(A, b)
    print(x)
    #print(A, "\n", b)
    