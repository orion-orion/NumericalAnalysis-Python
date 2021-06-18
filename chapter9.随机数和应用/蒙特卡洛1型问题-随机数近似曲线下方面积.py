'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-05-22 21:09:23
LastEditors: ZhangHongYu
LastEditTime: 2021-05-22 21:13:00
'''
#用随机数近似 y = x**2在[0, 1]下的面积

import random
sum_v = 0.0
n = 30
for i in range(n):
    ui = random.random() 
    sum_v += ui**2
print("近似面积为:", sum_v/n)


# 用随机数近似