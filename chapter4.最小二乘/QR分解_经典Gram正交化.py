'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-06-22 09:45:36
LastEditors: ZhangHongYu
LastEditTime: 2021-06-22 10:26:58
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

if __name__ == '__main__':
    # 前提: A中列向量线性无关
    A = np.array(
        [
            [1, -4],
            [2, 3],
            [2, 2]
        ]
    )
    Q, R = QR_Gram_Schmidt(A)
    print(Q, "\n\n", R)
