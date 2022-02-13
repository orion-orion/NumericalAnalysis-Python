'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-06-20 15:09:04
LastEditors: ZhangHongYu
LastEditTime: 2021-06-22 14:42:23
'''
import numpy as np
if __name__ == '__main__':
    
    x = np.array([(2+i/5) for i in range(11)]).reshape(-1, 1)
    y = 1 + x + x**2 + x**3 + x**4 + x**5 + x**6 + x**7 #(1被广播到向量的每个维度)
    A = np.concatenate([x**i for i in range(8)], axis=1)
    # print(A)
    # AT_A = A.T.dot(A)
    # AT_y = A.T.dot(y)
    # c_bar = np.linalg.solve(AT_A, AT_y)
    # print("最小二乘估计得到的参数:", c_bar)
    # # 条件数
    # print("条件数:", np.linalg.cond(AT_A))

    Q, R = np.linalg.qr(A, mode="complete")
    # 最小二乘误差||e||2 = ||(0, 0, 3)||2 = 3
    e = Q.T.dot(y)[-1]
    print("最小二乘误差", e[0], "\n")
    # 注意在解方程的时候一定要使“上部分”相等
    x = np.linalg.solve(R[:A.shape[1], :], Q.T.dot(y)[:A.shape[1]])
    print("最小二乘解\n:", x)
    print("条件数:", np.linalg.cond(R.T.dot(R)))
