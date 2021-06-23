'''
Descripttion: 算法证明见Golub, Van Loan《矩阵计算》
Version: 1.0
Author: ZhangHongYu
Date: 2021-06-22 21:00:46
LastEditors: ZhangHongYu
LastEditTime: 2021-06-23 08:33:57
'''
import numpy as np
# 对比
# def nsi(A, k):
#     # Q0 = I
#     # eye为方阵接收一个参数n
#     Q = np.eye(A.shape[0]) 
#     for i in range(k):
#         # 乘A, qr分解将其正交化后再乘A，以此迭代，这种写法很简洁，紧致
#         Q, R = np.linalg.qr(A.dot(Q), mode="reduced")
#     # 瑞利商
#     # 对角线元素即为所有特征值
#     lams = np.diag(Q.T.dot(A).dot(Q))
#     return Q, lams 

# 归一化同时迭代，k是迭代步数
# 欲求A特征值，A肯定是方阵
def unshiftedqr(A, k):
    # Q0 = I
    # eye为方阵接收一个参数n
    Q = np.eye(A.shape[0]) 
    Qbar = Q.copy()
    R = A.copy()
    for i in range(k):
        # QR分解
        Q, R = np.linalg.qr(R.dot(Q), mode="reduced")
        # 累计Q
        Qbar = Qbar.dot(Q)

    # 对角线收敛到特征值
    lams = np.diag(R.dot(Q))
    # 输出特征值lams和特征向量矩阵Q_bar
    return Qbar, lams 
if __name__ == '__main__':
    A = np.array(
        [
            [1, 3],
            [2, 2]
        ]
    )
    k = 10
    Q, lams = unshiftedqr(A, k)

    # 特征向量为(-0.707, -0.707)，(-0.707, 0.707)
    # 特征值是4和-1
    # 占优特征值为4，占优特征向量为(-0.707, -0.707)，与幂迭代结果一致
    # 幂迭代占优特征向量为(0.7, 0.7)，占优特征值为4
    # 真实客观情况：占优特征向量(1, 1)，占优特征值为4
    # 一个特征值对应的特征向量(组)可以随意缩放长度，其实是无数个
    print(Q, "\n\n", lams)

