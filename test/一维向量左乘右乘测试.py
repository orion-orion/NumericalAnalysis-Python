'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-06-22 15:47:30
LastEditors: ZhangHongYu
LastEditTime: 2021-06-22 15:47:38
'''
a = np.array([1, 2])
A = np.array([
    [2, 3],
    [5, 6]
])
# 返回[[12 15]]
print(a.reshape(1, -1).dot(A))
# 返回[12 15]，说明一维向量的维度可以同时用于左乘和右乘
print(a.dot(A))
    