'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-06-23 21:25:39
LastEditors: ZhangHongYu
LastEditTime: 2021-06-28 19:10:19
'''
import numpy as np
def svd(A):
    eigen_values, eigen_vectors = np.linalg.eig(A.T.dot(A))
    singular_values = np.sqrt(eigen_values)
    singular_values = list(reversed(singular_values))
    # 由等式Avi = siui(vi是右奇异向量, ui是左奇异向量)
    # 依次计算左奇异向量
    U = np.zeros((A.shape[0], A.shape[1]))
    for i in range(A.shape[1]):
        if singular_values[i] != 0:
            u = A.dot(eigen_vectors[:, i])/singular_values[i]
        # else:
        #     u =  np.zeros(A.shape[0])
        #     u[i] = 1
        U[:, i] = u
    # 给U加上标准正交基去构造R3的基
    for i in range(A.shape[1], A.shape[0]):
        basis = np.zeros((A.shape[0], 1))
        basis[i] = 1
        U = np.concatenate([U, basis], axis=1)
    S = np.diag(singular_values)
    S = np.concatenate([S, np.zeros((A.shape[0]-A.shape[1], A.shape[1]))], axis=0)
    return U, S, eigen_vectors

if __name__ == '__main__':
    # 例一：非方阵
    # A = np.array(
    #     [
    #         [0, -1/2],
    #         [3, 0],
    #         [0, 0]
    #     ]
    # )
    # 例二：方阵
    # A = np.array(
    #    [
    #        [0, 1],
    #        [0, -1]
    #    ]
    # )
    # 例三：对称矩阵
    A = np.array(
        [
            [0, 1],
            [1, 3/2]
        ]
    )
    U, S, V = svd(A)
    print(U, "\n\n", S, "\n\n", V)

    print("\n\n\n")

    # 调用api核对
    U2, S2, V2 = np.linalg.svd(A)
    print(U2, "\n\n", S2, "\n\n", V2)
