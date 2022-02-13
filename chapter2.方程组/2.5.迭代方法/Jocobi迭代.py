'''
Descripttion: 小型矩阵向量化未必比naive快，但大型矩阵向量化远超naive
Version: 1.0
Author: ZhangHongYu
Date: 2021-03-08 17:26:11
LastEditors: ZhangHongYu
LastEditTime: 2021-05-30 16:30:09
'''
import numpy as np
import time
eps = 1e-6
n = 6 #迭代次数
#向量化实现
def Jocobi(A, b):
    assert(A.shape[0] == A.shape[1] == b.shape[0])
    x = np.zeros(b.shape, dtype=np.float32)
    d = np.diag(A) #diag即可提取A的对角线元素，也可构建对角阵
    R = A - np.diag(d) #r为余项
    # U = np.triu(R)   #如果想获取不含对角线的L和U需如此，直接np.triu()得到的是含对角线的
    # L = np.tril(R)
    # 迭代次数
    for t in range(n):
        x = (b-np.matmul(R, x))/d
    return x    

#普通实现
def Jocobi_naive(A, b):
    assert(A.shape[0] == A.shape[1] == b.shape[0])
    x = np.zeros(b.shape, dtype=np.float32)
    # 迭代次数
    for t in range(n):
        #普通实现
        for i in range(x.shape[0]):
            val = b[i]
            for j in range(A.shape[1]):
                if j != i :
                    val -= A[i, j] * x[j]
            x[i] = val/A[i][i]
    return x    

if __name__ == '__main__':
    # A一定要是主对角线占优矩阵
    # A = np.array(
    #     [
    #         [3, 1, -1],
    #         [2, 4, 1],
    #         [-1, 2, 5]
    #     ],dtype=np.float32
    # )
    A = np.eye(1000, dtype=np.float32)

    # b = np.array(
    #     [4, 1, 1],dtype=np.float32
    # )
    b = np.zeros((1000,), np.float32)

    start1 = time.time()
    x1 = Jocobi_naive(A, b)
    end1 = time.time()
    print("time: %.10f" % (end1-start1))
    print(x1)


    start2 = time.time()
    x2 = Jocobi(A, b)
    end2 = time.time()
    print("time: %.10f" % (end2 - start2))
    print(x2)
    #print(A, "\n", b)
    