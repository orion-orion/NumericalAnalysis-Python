'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-06-23 15:42:51
LastEditors: ZhangHongYu
LastEditTime: 2022-05-31 21:15:46
'''
import numpy as np
# 归一化同时迭代，k是迭代步数
# 欲推往A特征值的方向，A肯定是方阵
def PageRank(A, p, k, q):
    assert(A.shape[0]==A.shape[1])
    n = A.shape[0]
    M = A.T.astype(np.float32) #注意要转为浮点型
    for i in range(n):
        M[:, i] = M[:, i]/np.sum(M[:, i])
    G = (q/n)*np.ones((n,n)) + (1-q)*M
    #G_T = G.T
    p_t = p.copy()
    for i in range(k):
        y = G.dot(p_t)
        p_t = y/np.max(y)
    return p_t/np.sum(p_t)
if __name__ == '__main__':
    A = np.array(
        [
            [0, 1, 1],
            [0, 0, 1],
            [1, 0, 0]
        ]
    )
    k = 20
    p = np.array([1, 1, 1])
    q = 0.15 #概率为1移动到一个随机页面通常为0.15
    # 概率为1-q移动到与本页面链接的页面
    R= PageRank(A, p, k, q)
    print(R)
