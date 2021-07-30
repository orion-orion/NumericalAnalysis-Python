'''
Descripttion: 将瑞利商做为逆向幂迭代的特征值加速收敛
Version: 1.0
Author: ZhangHongYu
Date: 2021-06-22 16:55:06
LastEditors: ZhangHongYu
LastEditTime: 2021-07-24 20:20:13
'''
import numpy as np

def Ray(A, x, k):
    for j in range(k):
        # 为了让数据不失去控制
        # 每次迭代前先对x进行归一化
        u = x/np.linalg.norm(x)
        
        #用瑞利商表示特征值的近似值lam
        lam = u.dot(A).dot(u)

        # 求解(A-lamj-1 I)Xj = uj-1
        # 收敛后A-lamj-1 I为奇异矩阵，不能继续迭代
        # 故在这种情况出现之前终止迭代
        try:
            # 单位矩阵I用np.eye实现
            x = np.linalg.solve(A - lam*np.eye(A.shape[0]), u) #逆向幂迭代
        except  np.linalg.LinAlgError as e:
            print("A-lam I 为奇异矩阵，停止迭代!")
            break

    # 最后一次迭代得到的特征向量x需要归一化为u
    u = x / np.linalg.norm(x)
    # 瑞利商(这里容易漏掉)
    lam = u.dot(A).dot(u)
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
    u, lam = Ray(A, x, k)
    # u为 [0.70710341 0.70711015]，指向特征向量[1, 1]的方向
    print("占优的特征向量:\n", u)
    print("占优的特征值:\n", lam)