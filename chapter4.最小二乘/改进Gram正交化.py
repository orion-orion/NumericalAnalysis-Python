'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-06-22 09:45:36
LastEditors: ZhangHongYu
LastEditTime: 2021-06-22 15:15:21
'''
import numpy as np
def QR_Gram_Schmidt(A):
    R = np.zeros((A.shape[1], A.shape[1]))
    Q = np.zeros(A.shape)
    for j in range(A.shape[1]):
        # 默认copy为深拷贝，此处y为A的一维向量切片
        y = A[:, j].copy()   
        # 生成R中对角线以上的元素
        for i in range(j):
            R[i, j] = Q[:, i].dot(A[:, j])
            y = y - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(y)
        Q[:, j] = y/R[j, j] 
    return Q, R

def Modified_QR_Gram_Schmidt(A):
    R = np.zeros((A.shape[1], A.shape[1]))
    Q = np.zeros(A.shape)
    for j in range(A.shape[1]):
        # 默认copy为深拷贝，此处y为A的一维向量切片
        y = A[:, j].copy()   
        # 生成R中对角线以上的元素
        for i in range(j):
            # 这里直接算的是qi在y上的投影，而不是之前的Aj
            # y是已经减掉投影后的
            R[i, j] = Q[:, i].dot(y) 
            y = y - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(y)
        Q[:, j] = y/R[j, j] 
    return Q, R
epsilon = 10e-10

if __name__ == '__main__':
    # 前提: A中列向量线性无关
    # A = np.array(
    #     [
    #         [1, -4],
    #         [2, 3],
    #         [2, 2]
    #     ]
    # )
    # Q, R = Modified_QR_Gram_Schmidt(A)
    # print(Q, "\n\n", R)
    # 前提: A中列向量线性无关
    A = np.array(
        [
            [1, 1, 1],
            [epsilon, 0, 0],
            [0, epsilon, 0],
            [0, 0, epsilon]
        ]
    )

    Q1, R1 = Modified_QR_Gram_Schmidt(A)
    Q, R = QR_Gram_Schmidt(A)
    print("改进施密特方法的Q和R:")
    print(Q1, "\n\n", R1)
    print("经典施密特方法的Q和R:")
    print(Q, "\n\n", R)
    print("改进施密特方法条件数:\n")
    print(np.linalg.cond(Q1))
    print("经典施密特方法条件数:\n")
    print(np.linalg.cond(Q))
    