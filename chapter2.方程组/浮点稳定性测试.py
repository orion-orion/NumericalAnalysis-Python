'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-05-29 20:57:38
LastEditors: ZhangHongYu
LastEditTime: 2021-05-29 21:37:56
'''
import numpy as np
b = np.array([[1]]) #默认你这个矩阵里存的就是整形
b[0] = b[0] - 6.99999976  #这里b[0] = b[0, 0]，单元素列表赋值自动取元素
print(b[0]) #自动向上取整截断为-5
