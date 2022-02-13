'''
Descripttion: 逆向幂迭代
Version: 1.0
Author: ZhangHongYu
Date: 2021-06-22 16:55:06
LastEditors: ZhangHongYu
LastEditTime: 2021-06-22 17:18:21
'''
import numpy as np

# 幂迭代本质上每步进行归一化的不动点迭代
def powerit(A, x, s, k):
    As = A-s*np.eye(A.shape[0])
    for j in range(k):
        # 为了让数据不失去控制
        # 每次迭代前先对x进行归一化
        u = x/np.linalg.norm(x)
        
        # 求解(A-sI)Xj = uj-1
        x = np.linalg.solve(As, u)
        lam = u.dot(x)
    lam = 1/lam + s
        
    # 最后一次迭代得到的特征向量x需要归一化为u
    u = x / np.linalg.norm(x)
    return u, lam        

if __name__ == '__main__':
    A = np.array(
        [
            [1, 3],
            [2, 2]
        ]
    )
    x = np.array([-5, 5])
    k = 10
    # 逆向幂迭代的平移值，可以通过平移值收敛到不同的特征值
    s = 2 
    # 返回占优特征值和对应的特征值
    u, lam = powerit(A, x, s, k)
    # u为 [0.70710341 0.70711015]，指向特征向量[1, 1]的方向
    print("占优的特征向量:\n", u)
    print("占优的特征值:\n", lam)