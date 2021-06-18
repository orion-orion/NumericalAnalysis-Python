'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-05-29 22:47:06
LastEditors: ZhangHongYu
LastEditTime: 2021-05-30 10:33:25
'''
import numpy as np
import time
import dis

# 对于Python而言对然两个函数字节码不一样，但缓存是透明的，故执行速度一样
# C++是row比col快
# 对于Java而言row_gaxpy得到优化反而比C++快，但col_gaxpy就比C++慢了
# 但无论如何Java和C++都比Python快了
def row_gaxpy(A, x):
    y = np.zeros((A.shape[0],), dtype=np.float32)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            y[i] += A[i, j] * x[j]
    return y

def col_gaxpy(A, x):
    y = np.zeros((A.shape[0],), dtype=np.float32)
    for j in range(A.shape[0]):
        for i in range(A.shape[1]):
            y[i] += A[i, j] * x[j]
    return y

# A = np.random.rand(5000, 5000)
# x = np.random.rand(5000)

# begin = time.time()
# y1 = row_gaxpy(A, x)
# end = time.time()
# print("time: %.1f" % (end - begin))

# begin = time.time()
# y2 = col_gaxpy(A, x)
# end = time.time()
# if np.array_equal(y1, y2):
#     print("equal!")
# print("time: %.1f" % (end - begin))

print(dis.dis(row_gaxpy))
print("*************")
print(dis.dis(col_gaxpy))