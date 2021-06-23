'''
Descripttion:为了允许计算复数特征值，必须允许在实数舒尔形式的对角线上存在2*2的块 
Version: 1.0
Author: ZhangHongYu
Date: 2021-06-23 10:17:21
LastEditors: ZhangHongYu
LastEditTime: 2021-06-23 11:21:05
'''
import numpy as np
'''
@Description: 
@Param: 矩阵A
@Return: 特征值lam
@Author: ZhangHongYu
@param {*} A
'''
def shiftedqr0(A):
    tol = 1e-14
    m = A.shape[0]  
    lam = np.zeros((A.shape[0], ))
    n = m
    while n > 1:
        while max(abs(A[n-1, :n])) > tol:
            # 定义平移mu
            mu = A[n-1, n-1]
            Q, R = np.linalg.qr(A-mu*np.eye(n))
            A = R.dot(Q) + mu * np.eye(n)
        # 声明特征值
        lam[n-1] = A[n-1, n-1]
        # 降低n
        n = n - 1
        # 收缩
        A = A[:n, :n] #.copy()
    lam[0] = A[0, 0]
    return lam

if __name__ == '__main__':
    A = np.array(
        [
            [1, 3],
            [2, 2]
        ]
    )
    k = 10
    Q, lams = shiftedqr0(A)

    # 特征向量为(-0.707, -0.707)，(-0.707, 0.707)
    # 特征值是4和-1
    # 占优特征值为4，占优特征向量为(-0.707, -0.707)，与幂迭代结果一致
    # 幂迭代占优特征向量为(0.7, 0.7)，占优特征值为4
    # 真实客观情况：占优特征向量(1, 1)，占优特征值为4
    # 一个特征值对应的特征向量(组)可以随意缩放长度，其实是无数个
    print(Q, "\n\n", lams)
    