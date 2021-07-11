'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-07-02 20:26:12
LastEditors: ZhangHongYu
LastEditTime: 2021-07-11 16:44:01
'''
import numpy as np
import math
def gss(f, a, b, k):
    g = (math.sqrt(5)-1)/2
    # 计算x1和x2
    x1 = a + (1-g)*(b-a)
    x2 = a + g*(b-a)
    f1, f2 = f(x1), f(x2)
    for i in range(k):
        if f1 < f2 :
            # 依次更新b, x2, x1
            b = x2
            x2 = x1
            # 这里代码设计的很巧妙，b是已经更新后的新b
            x1 = a + (1-g)*(b-a)
            f2 = f1
            f1 = f(x1)
        else:
            a = x1
            x1 = x2
            x2 = a + g*(b-a)
            f1 = f2
            f2 = f(x2)
    y = (a+b)/2    
    return(a, b), y
if __name__ == '__main__':
    a, b = 0, 1
    k = 15
    (a,b), y = gss(lambda x: x**6-11*x**3+17*x**2-7*x+1, a, b, k)
    print("(%.4f, %.4f)"%(a, b), y)