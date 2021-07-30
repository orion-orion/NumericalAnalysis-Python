'''
Descripttion: PA=LU

Version: 1.0
Author: ZhangHongYu
Date: 2021-05-22 21:53:43
LastEditors: ZhangHongYu
LastEditTime: 2021-07-26 12:21:27
'''
import numpy as np
from copy import deepcopy
eps = 1e-6

# 消去第j个主元时，将第j列中最大的元素行和主元所在行交换位置
# 以保证乘子A[i, j]/A[j, j]为小数
def check_and_permute(P, A, j, b):
    max_row = j
    max_val = A[j, j]
    #每次置换对应左乘一个置换矩阵
    P_plus = np.eye(A.shape[0], dtype=np.float32) #初始化要更新的左乘置换矩阵
    for i in range(j+1, A.shape[0]):
        if abs(A[i, j]) > abs(max_val):
            max_val = A[i, j]
            max_row = i
    # 交换max_row行和主元所在的第j行
    # temp = deepcopy(A[max_row, :])
    # A[max_row, :] = deepcopy(A[j, :])
    # A[j, :] = temp

    # temp = deepcopy(b[max_row, :])
    # b[max_row, :] = deepcopy(b[j, :])
    # b[j, :] = temp
    # deepcopy()必须要，否则temp只是一个引用，改temp后A也会变
    A[j, :], A[max_row, :] = deepcopy(A[max_row, :]), deepcopy(A[j, :])
    b[j, :], b[max_row, :] = deepcopy(b[max_row, :]), deepcopy(b[j, :])
    #如果主元本身最大，即max_row = j，那相当于乘一个单位阵
    P_plus[j, j], P_plus[max_row, max_row] = 0, 0
    P_plus[j, max_row], P_plus[max_row, j] = 1, 1
    #print(P_plus)
    print("当次置换的矩阵为:", P_plus)
    P = np.matmul(P_plus, P)
    return P

# 消去步骤
def PA_LU_decomposition(A): #假设A是方阵，A.shape[0] == A.shape[1], python中变量是传拷贝，数组等对象是传引用
    assert(A.shape[0] == A.shape[1])
    U = deepcopy(A)
    L = np.zeros(A.shape, dtype=np.float32)
    P = np.eye(A.shape[0], dtype=np.float32) #初始化置换矩阵为单位阵I
    for j in range(U.shape[1]): #消去第j列的数
        # abs(U[j ,j])为要消去的主元
        print("置换前的U：", U)
        P = check_and_permute(P, U, j, b)
        print("置换后的U", U)
        L[j, j] = 1
        # 消去主对角线以下的元素A[i, j]
        for i in range(j+1, U.shape[0]):
            mult_coeff = U[i, j]/U[j, j]
            L[i, j] = mult_coeff
            # 对这A中这一行都进行更新
            for k in range(j, U.shape[1]):
                U[i, k] = U[i, k] - mult_coeff * U[j, k]
    return P, L, U

#常规的上三角进行回代(此例中对角线不为0)
def gaussion_putback_U(A, b):
    x = np.zeros((A.shape[0], 1), dtype=np.float32) #注意float类型，否则又要截断成0
    for i in reversed(range(A.shape[0])): #算出第i个未知数
        for j in range(i+1, A.shape[1]):
            b[i] = b[i] - A[i, j] * x[j]
        x[i] = b[i] / A[i, i]
    return x

#下三角进行回代(此例中对角线不为0)
def gaussion_putback_L(A, b):
    x = np.zeros((A.shape[0], 1), dtype=np.float32)  #注意float类型，否则又要截断成0
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

def PA_LU_putback(P, L, U, b):
 
    #  LUx = Pb ，令Ux = c
    # 解 Lc = Pb
    # 先要对b也进行置换
    b = np.matmul(P, b) #b存为矩阵的好处在此时就能体现

    c = gaussion_putback_L(L, b) #上三角回代
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
    # PA=LU分解后对b也要置换
    # 还是尽量使消元与回代分离
    
    # 分解步骤
    # Ax = b => PAx = Pb => PA = LU
    P, L, U = PA_LU_decomposition(A)
    print(P)
    print(L)
    print(U)
    # 我感觉P, L, U都计算正确，但是x回代计算的结果就是不对
    # 回代步骤，要用P对b也进行置换
    x = PA_LU_putback(P, L, U, b)
    print(x)
    #print(A, "\n", b)