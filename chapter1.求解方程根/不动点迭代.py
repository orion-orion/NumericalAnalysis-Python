'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-03-08 15:13:40
LastEditors: ZhangHongYu
LastEditTime: 2021-03-08 15:21:11
'''
import numpy as np
import math
def fpi(g, x0 ,k): #迭代k次,包括x0在内共k+1个数
    x = np.zeros(k+1,)
    x[0] = x0
    for i in range(1, k+1):
        x[i] = g(x[i-1])
    return x[k]
if __name__ == '__main__':
    res = fpi(lambda x: math.cos(x), 0, 10)
    print(res)