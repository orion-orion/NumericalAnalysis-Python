'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-06-22 09:45:36
LastEditors: ZhangHongYu
LastEditTime: 2021-06-22 11:25:20
'''
import numpy as np


if __name__ == '__main__':
    # 前提: A中列向量线性无关
    A = np.array(
        [
            [1, -4],
            [2, 3],
            [2, 2]
        ]
    )
    b = np.array([-3, 15, 9])
    Q, R = np.linalg.qr(A, mode="complete")
    # 最小二乘误差||e||2 = ||(0, 0, 3)||2 = 3
    e = Q.T.dot(b)[-1]
    print("最小二乘误差", e)
    # 注意在解方程的时候一定要使“上部分”相等
    x = np.linalg.solve(R[:A.shape[1], :], Q.T.dot(b)[:A.shape[1]])
    print(x)
