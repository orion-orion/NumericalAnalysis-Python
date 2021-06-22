'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-06-22 09:45:36
LastEditors: ZhangHongYu
LastEditTime: 2021-06-22 10:55:51
'''
import numpy as np
def Complete_QR_Gram_Schmidt(A):
    # R只是大小扩充了，剩余部分仍然是0没有计算
    R = np.zeros(A.shape)
    # Q的剩余部分用dummy向量An+1, An+2, ..., Am计算
    Q = np.zeros((A.shape[0], A.shape[0]))
    # 先找到正交单位向量q1, q2, ..., qn
    for j in range(A.shape[1]):
        # 默认copy为深拷贝，此处y为A的一维向量切片
        y = A[:, j].copy()   
        # 生成R中对角线以上的元素
        for i in range(j):
            R[i, j] = Q[:, i].dot(A[:, j])
            y = y - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(y)
        Q[:, j] = y/R[j, j] 

    # 使用扩充的dummy向量An+1, An+2, ..., Am
    # 来扩充找到qn+1, qn+2, ..., qm，对于扩充的Q没有对应的r值
    for j in range(A.shape[1], A.shape[0]): 
        # 加上处于第 j + 1 位置的dummy向量Aj，
        # 一般是(1, 0, 0), (0, 1, 0)...
        A_plus = np.zeros((A.shape[0],))
        A_plus[j-A.shape[1]] = 1
        y = A_plus.copy()
        for i in range(j):
            y =  y - Q[:, i].dot(A_plus)*Q[:, i]
        Q[:, j] = y/np.linalg.norm(y)

    return Q, R

if __name__ == '__main__':
    # 前提: A中列向量线性无关
    A = np.array(
        [
            [1, -4],
            [2, 3],
            [2, 2]
        ]
    )
    Q, R = Complete_QR_Gram_Schmidt(A)
    print(Q, "\n\n", R)
