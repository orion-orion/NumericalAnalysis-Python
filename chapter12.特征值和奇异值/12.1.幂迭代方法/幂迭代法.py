'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-06-22 15:38:31
LastEditors: ZhangHongYu
LastEditTime: 2021-07-08 17:19:23
'''
import numpy as np

# 幂迭代本质上每步进行归一化的不动点迭代
# 毕竟x左乘一个矩阵A相当于g(*)作用于x
def powerit(A, x, k):
    for j in range(k):
        # 为了让数据不失去控制
        # 每次迭代前先对x进行归一化
        u = x/np.linalg.norm(x)
        # 计算下一轮x，即将u乘以A, 为Auj-1
        x = A.dot(u)
        # 欲用最小二乘解特征方程x*lamb = Ax
        # x是特征向量的近似，lamb未知，系数矩阵是x，参数是lamb，b为Ax
        # 正规(法线)方程指出最小二乘解为:xT_x*lamb = xT(Ax)的解(方程两边同时乘AT的逆消掉)
        # 或者lamb = xTAx/xTx
        # 即lamb = uTAu，u.dot(A).dot(u) (又X=Au，避免重复计算这里乘x)
        # 故如下计算出本轮对应的特征值
        lam = u.dot(x)
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
    # 返回占优特征值和对应的特征值
    u, lam = powerit(A, x, k)
    # u为 [0.70710341 0.70711015]，指向特征向量[1, 1]的方向
    print("占优的特征向量:\n", u)
    print("占优的特征值:\n", lam)

