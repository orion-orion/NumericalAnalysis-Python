'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-07-01 21:35:04
LastEditors: ZhangHongYu
LastEditTime: 2021-10-16 15:34:09
'''
import numpy as np
from sklearn.decomposition import PCA
def approximation(A, p):
    U, s, V_T = np.linalg.svd(A)
    B = np.zeros(A.shape)
    for i in range(p):
        B += s[i]*U[:,i].reshape(-1, 1).dot(V_T[i, :].reshape(1, -1))
    return B

if __name__ == '__main__':
    # 例一：
    A = np.array(
        [
            [0, 1],
            [1, 3/2],
        ]
    )
    # 例二：
    # A = np.array(
    #     [
    #         [3, 2, -2, -3],
    #         [2, 4, -1, -5]
    #     ]
    # )

    # p为近似矩阵的秩，秩p<=r
    p = 1
    B = approximation(A, p)
    print(B)

    #可以看到最终得到的矩阵秩为1
    print(np.linalg.matrix_rank(B))








    
    # 调用api核对，和传统PCA比较
    # pca= PCA(n_components=2, svd_solver='auto')
    # B2 = pca.fit_transform(A)
    # print(B2)