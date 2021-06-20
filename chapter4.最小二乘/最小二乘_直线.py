'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-06-20 11:37:16
LastEditors: ZhangHongYu
LastEditTime: 2021-06-20 15:38:27
'''
import numpy as np
if __name__ == '__main__':
    X = np.array(
        [
            [-1],
            [0],
            [1],
            [2]
        ]
    )
    y = [1, 0, 0, -2]
    # 多项式拟合的话要先对X预处理，从第三列开始依次计算出第二列的次方值(还是拟合平面上的点，不过扩充了)
    # 此处X一共两列，最高次数只有1次
    A = np.concatenate([np.ones([X.shape[0], 1]), X], axis=1)
    AT_A = A.T.dot(A)
    AT_y = A.T.dot(y)
    c_bar = np.linalg.solve(AT_A, AT_y)
    print("最小二乘估计得到的参数:", c_bar)
    # 条件数
    print("条件数:", np.linalg.cond(AT_A))
