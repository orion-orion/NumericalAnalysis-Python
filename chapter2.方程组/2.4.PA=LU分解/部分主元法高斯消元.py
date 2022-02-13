'''
Descripttion: 部分主元法可解决淹没问题和0主元问题，高斯消元都考虑非奇异阵（肯定是方的）
Version: 1.0
Author: ZhangHongYu
Date: 2021-05-29 15:49:30
LastEditors: ZhangHongYu
LastEditTime: 2021-05-29 21:05:20
'''
import numpy as np
from copy import deepcopy
eps = 1e-6
# 消去第j个主元时，将第j列中最大的元素行和主元所在行交换位置
# 以保证乘子A[i, j]/A[j, j]为小数
def check_and_permute(A, j, b):
    max_row = j
    max_val = A[j, j]
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
# 消去步骤
def gaussion_elimination(A, b): #假设A是方阵，A.shape[0] == A.shape[1], python中变量是传拷贝，数组等对象是传引用
    assert(A.shape[0] == A.shape[1])
    for j in range(A.shape[1]): #消去第j列的数，即使用第j个主元
        print(A)
        check_and_permute(A, j, b)
        # 消去主对角线以下的元素A[i, j]
        for i in range(j+1, A.shape[0]):
            mult_coeff = A[i, j]/A[j, j] # 保证 0 < abs(multi_coeff) < 1，且主元系数A[j, j]决不为0
            # 对这A中这一行都进行更新
            for k in range(j, A.shape[1]):
                A[i, k] = A[i, k] - mult_coeff * A[j, k]
            b[i] = b[i] - mult_coeff * b[j] #二维的b取单dim1的索引即[1]这种含单个元素的列表

def gaussion_putback(A, b):
    x = np.zeros((A.shape[0], 1))
    for i in reversed(range(A.shape[0])): #算出第i个未知数
        for j in range(i+1, A.shape[1]):
            b[i] = b[i] - A[i, j] * x[j]
        x[i] = b[i] / A[i, i]  #主元系数A[i, i]决不应为0，否则出错
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
        dtype=np.float32
    )
    gaussion_elimination(A, b)
    x = gaussion_putback(A, b)
    print(x)
    #　print(A, "\n", b)
    #  想答案是这样:
    #     [[3.        ]
    #  [1.00000012]
    #  [2.00000048]]
    # 需要换成float64提高精度或直接用decimal应对