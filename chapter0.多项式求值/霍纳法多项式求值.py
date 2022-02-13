'''
Descripttion: horn法则多项式求值
Version: 1.0
Author: ZhangHongYu
Date: 2021-03-07 11:40:20
LastEditors: ZhangHongYu
LastEditTime: 2021-03-07 11:59:29
'''
import numpy as np

def poly(order, coeff, x, basis):
    if basis is None:
        basis = np.zeros((order,), dtype=np.float32)
    # 初始递推值
    y = coeff[-1]
    for i in reversed(range(order)):
        y = y * (x - basis[i]) + coeff[i]
    return y

if __name__ == '__main__':
    # 令所有基点为0的普通多项式
    res1 = poly(4, np.array([-1, 5,-3, 3, 2]), 1/2, np.array([0, 0, 0, 0]))
    print(res1)
    # 基点不为0的三阶插值多项式
    res2 = poly(3, np.array([1, 1/2, 1/2, -1/2]), 1, [0, 2, 3])
    print(res2)