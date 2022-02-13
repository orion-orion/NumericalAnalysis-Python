'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-06-23 21:25:39
LastEditors: ZhangHongYu
LastEditTime: 2021-10-15 21:56:23
'''
import numpy as np
def svd(A):
    eigen_values, eigen_vectors = np.linalg.eig(A.T.dot(A))
    singular_values = np.sqrt(eigen_values)
    #这里奇异值要从大到小排序，特征向量也要随之从大到小排
    val_vec = [] #存储奇异值-特征向量对
    for i in range(len(eigen_values)):
        val_vec.append((singular_values[i], eigen_vectors[:, i]))
    val_vec.sort(key = lambda x:-x[0])
    singular_values = [ pair[0] for pair in val_vec]
    eigen_vectors = [ pair[1] for pair in val_vec]

    # 在计算左奇异向量之前，先要对右奇异向量
    # 也就是特征向量组成的基正交化，不过linalg.eig返回的是已经正交化的

    # 由等式Avi = siui(vi是右奇异向量, ui是左奇异向量)
    # 依次计算左奇异向量
    U = np.zeros((A.shape[0], A.shape[1]))
    for i in range(A.shape[1]):
        if singular_values[i] != 0:
            u = A.dot(eigen_vectors[i])/singular_values[i]
        else:
            # u =  与之前的u正交的向量
            pass #这里还没写好
        U[:, i] = u
    # 给U加上标准正交基去构造R3的基
    for i in range(A.shape[1], A.shape[0]):
        basis = np.zeros((A.shape[0], 1))
        basis[i] = 1
        U = np.concatenate([U, basis], axis=1)
    # S = np.diag(singular_values)
    # S = np.concatenate([S, np.zeros((A.shape[0]-A.shape[1], A.shape[1]))], axis=0)
    eigen_vectors = [vec.reshape(-1, 1) for vec in eigen_vectors]
    eigen_vectors = np.concatenate(eigen_vectors, axis=1)
    return U, singular_values, eigen_vectors

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
    print("我们实现的算法结果：")
    print(U, "\n", S, "\n", V)
    print("\n")
    print("调用库函数的计算结果：")
    # 调用api核对
    U2, S2, V2 = np.linalg.svd(A)
    print(U2, "\n", S2, "\n", V2)
