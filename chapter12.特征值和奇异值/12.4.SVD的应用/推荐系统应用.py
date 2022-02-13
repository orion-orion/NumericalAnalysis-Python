'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-07-09 09:07:46
LastEditors: ZhangHongYu
LastEditTime: 2021-10-16 10:43:47
'''
import numpy as np


if __name__ == '__main__':
    M = np.array(
        [
            [0, 4.5, 2.0, 0],
            [4.0, 0, 3.5, 0],
            [0, 5.0, 0, 2.0],
            [0, 3.5, 4.0, 1.0]
        ]
    )
    U, S, V_T = np.linalg.svd(M)
    k = 2 # 取前2个奇异值对应的隐向量
    # 分别打印物品向量和用户向量
    Vec_user, Vec_item = U[:,:k], V_T[:k, :].T
    print(Vec_user, "\n\n", Vec_item)
