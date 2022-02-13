'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-05-22 21:20:28
LastEditors: ZhangHongYu
LastEditTime: 2021-05-22 21:24:14
'''
import random
import matplotlib.pyplot as plt
# 随机游走步数
#一维随机游走
t = 10
w = 1
plots = [w]
for i in range(1, t+1):
    if random.random() > 0.5:
        w = w + 1
    else:
        w = w- 1
    plots.append(w)
plt.plot(range(t+1), plots)
plt.show()

    