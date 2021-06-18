'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-05-30 14:57:56
LastEditors: ZhangHongYu
LastEditTime: 2021-05-30 15:01:49
'''
import numpy as np
eps = 1e-6
n = 6
# 消去步骤
def Jocobi(A, b):
    assert(A.shape[0] == A.shape[1] == b.shape[0])
    x = np.zeros(b.shape, dtype=np.float32)
    # 迭代次数
    for t in range(n):
        for i in range(x.shape[0]):
            val = b[i]
            for j in range(A.shape[1]):
                if j != i :
                    val -= A[i, j] * x[j]
            x[i] = val/A[i][i]
    return x    


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


def SOR(A, b, w):
    assert(A.shape[0] == A.shape[1] == b.shape[0])
    x = np.zeros(b.shape, dtype=np.float32)
    # 迭代次数
    for t in range(n):
        for i in range(x.shape[0]):
            val = b[i]
            for j in range(A.shape[1]):
                if j != i :
                    val -= A[i, j] * x[j]
            x[i] = (1-w)*x[i] + w*(val/A[i][i])
    return x    


if __name__ == '__main__':
    # A一定要是主对角线占优矩阵
    A = np.array(
        [
            [3, -1, 0, 0, 0, 1/2],
            [-1, 3, -1, 0, 1/2, 0],
            [0, -1, 3, -1, 0, 0],
            [0, 0, -1, 3, -1, 0],
            [0, 1/2, 0, -1, 3, -1],
            [1/2, 0, 0, 0, -1, 3]
        ],dtype=np.float32
    )
    b = np.array(
        [5/2, 3/2, 1, 1, 3/2, 5/2],dtype=np.float32
    )
    w = 1.1
    x1 = Jocobi(A, b)
    x2 = GaussSeidel(A, b)
    x3 = SOR(A, b, w)
    print(x1)
    print(x2)
    print(x3)
    #print(A, "\n", b)