'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-09-19 19:53:53
LastEditors: ZhangHongYu
LastEditTime: 2022-02-13 09:26:45
'''
import numpy as np
if __name__ == '__main__':
    x = np.array(
        [
            [-1],
            [0],
            [1],
            [2]
        ]
    )
    y = [1, 0, 0, -2]
    # 对数据矩阵x预处理，从第三列开始依次计算出第二列的次方值(还是拟合平面上的点，不过扩充了)
    # 此处A一共三列，最高次数有2次，即抛物线
    A = np.concatenate([np.ones([x.shape[0], 1]), x, x**2], axis=1)
    AT_A = A.T.dot(A)
    AT_y = A.T.dot(y)
    c_bar = np.linalg.solve(AT_A, AT_y) # 该API AT_y是一维/二维的都行
    print("最小二乘估计得到的参数:", c_bar)
    # 条件数
    print("条件数:", np.linalg.cond(AT_A))
