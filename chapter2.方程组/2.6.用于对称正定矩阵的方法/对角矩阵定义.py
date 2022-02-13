'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-06-06 16:45:50
LastEditors: ZhangHongYu
LastEditTime: 2021-06-06 16:55:19
'''
import numpy as np
n = 10
# 前者乃对角线往上数的第10阶
# 后者乃对角线往下数的第10阶(也就是-10阶)
A = np.diag(np.sqrt(range(1, n+1)))\
    + np.diag(np.cos(range(1, n-10+1)),10)\
        + np.diag(np.cos(range(1, n-10+1)), -10)    
print(A)