'''
Descripttion: 稀疏矩阵sparse库似乎只定义了稀疏矩阵.dot(普通向量)
以及一些稀疏矩阵的方法(triu,eye之类)
Version: 1.0
Author: ZhangHongYu
Date: 2021-03-08 17:26:11
LastEditors: ZhangHongYu
LastEditTime: 2021-05-30 17:10:42
'''
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import diags
import time
eps = 1e-6
n = 6 #迭代次数
#向量化普通实现
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

#向量化稀疏矩阵实现
def Jocobi_sparse(A:csr_matrix, b): #此处csr_matrix不会执行，仅仅参数提醒而已
    #assert(A.shape[0] == A.shape[1] == b.shape[0])
    x = np.zeros(b.shape, dtype=np.float32)
    d = np.diag(A) #diag即可提取A的对角线元素，也可构建对角阵
    R = A - np.diag(d) #r为余项
    R = csr_matrix(R) #采用(row, col，val)形式的元组存放
    # 迭代次数
    for t in range(n):
        x = (b-R.dot(x))/d
    return x  

if __name__ == '__main__':

    A = np.eye(10000, dtype=np.float32,) #可选择k让对角线偏移

    b = np.ones((10000,), np.float32)

    start1 = time.time()
    x1 = Jocobi(A, b)
    end1 = time.time()
    print("time: %.10f" % (end1-start1))
    print(x1)


    start2 = time.time()
    x2 = Jocobi_sparse(A, b)

    end2 = time.time()
    print("time: %.10f" % (end2 - start2))
    print(x2)

    