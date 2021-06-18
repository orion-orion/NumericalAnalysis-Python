'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-03-08 14:48:29
LastEditors: ZhangHongYu
LastEditTime: 2021-03-08 15:01:59
'''
import numpy as np
import math
def binary(f, a, b, tol):
    if f(a)*f(b) >=0 :
        raise ValueError("f(a)*f(b)<0 not satisfied!")
    while (b-a)/2 > tol:
        c=(a+b)/2 # 即使a和b是int,此处c自动转float了
        if f(c) == 0: #c是一个解，完成
            break
        if f(a)*f(c)<0 :
            b = c
        else:
            a = c
    return (a+b)/2
if __name__ == '__main__':
    res = binary(lambda x: x**3+x-1, 0, 1, 5*pow(10, -5))
    print(res)