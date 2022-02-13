'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-05-22 21:15:21
LastEditors: ZhangHongYu
LastEditTime: 2021-05-22 21:19:19
'''
import random
n= 10000
cnt = 0
for i in range(n):
    x = random.random() 
    y = random.random()
    if 4*(2*x-1)**4+8*(2*y-1)**8 < 1 + 2*(2*y-1)**3*(3*x-2)**2: #注意python连乘性和优先级
        cnt += 1
print("面积近似值为:", cnt/n)