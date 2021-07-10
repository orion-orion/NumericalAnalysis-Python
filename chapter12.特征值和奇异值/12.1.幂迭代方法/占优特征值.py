'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-06-22 15:38:31
LastEditors: ZhangHongYu
LastEditTime: 2021-07-08 17:05:41
'''
import numpy as np
def prime_eigen(A, x, k):
    x_t = x.copy()
    for j in range(k):
        x_t = A.dot(x_t)
    return x_t   
if __name__ == '__main__':
    A = np.array(
        [
            [1, 3],
            [2, 2]
        ]
    )
    x = np.array([-5, 5])
    k = 4
    r = prime_eigen(A, x, k)
    print(r)