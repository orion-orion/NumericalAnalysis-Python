'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-05-22 21:53:43
LastEditors: ZhangHongYu
LastEditTime: 2021-07-25 09:45:50
'''
import numpy as np
from copy import deepcopy
eps = 1e-6
# 消去步骤
def LU_decomposition(A): #假设A是方阵，A.shape[0] == A.shape[1], python中变量是传拷贝，数组等对象是传引用
    assert(A.shape[0] == A.shape[1])
    U = deepcopy(A)
    L = np.zeros(A.shape, dtype=np.float32)
    for j in range(U.shape[1]): #消去第j列的数
        # abs(U[j ,j])为要消去的主元
        if abs(U[j, j]) < eps:
            raise ValueError("zero pivot encountered!")  #无法解决零主元问题
            return
        L[j, j] = 1
        # 消去主对角线以下的元素A[i, j]
        for i in range(j+1, U.shape[0]):
            mult_coeff = U[i, j]/U[j, j]
            L[i, j] = mult_coeff
            # 对这A中这一行都进行更新
            for k in range(j, U.shape[1]):
                U[i, k] = U[i, k] - mult_coeff * U[j, k]
            
    return L, U

#常规的上三角进行回代(此例中对角线不为0)
def gaussion_putback_U(A, b):
    x = np.zeros((A.shape[0], 1), dtype=np.float32)
    for i in reversed(range(A.shape[0])): #算出第i个未知数
        for j in range(i+1, A.shape[1]):
            b[i] = b[i] - A[i, j] * x[j]
        x[i] = b[i] / A[i, i]
    return x

#下三角进行回代(此例中对角线不为0)
def gaussion_putback_L(A, b):
    x = np.zeros((A.shape[0], 1), dtype=np.float32)
    for i in range(A.shape[0]): #算出第i个未知数
        for j in range(i):
            b[i] = b[i] - A[i, j] * x[j]
            #草,如果b矩阵初始化时是整形，3-6.99999976 = ceil(-3.99999) = -3，
            # 直接给我向上取整(截断)约成整数了
            # if i == A.shape[0] - 1:
            #     print(A[i, j], "----", x[j], "----", A[i, j]*x[j])
            #     print(b[i])
        x[i] = b[i] / A[i, i]
    return x

def LU_putback(L, U, b):
    # Ax = b => LUx = b ，令Ux = c
    # 解 Lc = b
    c = gaussion_putback_L(L, b) #上三角回代
    print(c)
    # 再解 Ux = c
    x = gaussion_putback_U(U, c) #下三角回代
    return x

if __name__ == '__main__':
    A = np.array(
        [
            [1, 2, -1],
            [2, 1, -2],
            [-3, 1, 1]
        ],
        dtype=np.float32
    )
    b = np.array(
        [
            [3],
            [3],
            [-6]
        ],
        dtype=np.float32 #注意，此处必须是浮点型，否则整形的话后面就自动舍入了
    )
    # 单纯的LU分解过程不会对b有影响
    # 即消元与回代分离
    
    # 分解步骤
    L, U = LU_decomposition(A) # A=LU
    print(L)
    print(U)
    # 回代步骤
    x = LU_putback(L, U, b)
    print(x)
    #print(A, "\n", b)