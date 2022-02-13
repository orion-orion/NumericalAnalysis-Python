'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-05-22 20:57:46
LastEditors: ZhangHongYu
LastEditTime: 2021-05-22 21:04:04
'''
seed = 1
n = 100 #生成100个随机数
x = seed #初始化整数种子,不算在随机数内
a = int(pow(7, 5))
m = int(pow(2, 31))-1
for i in range(n):
    x = a * x % m
    ui = x/m
    print(x, "---", ui)

