'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-05-30 14:54:36
LastEditors: ZhangHongYu
LastEditTime: 2021-05-30 16:31:41
'''
'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-05-30 14:48:57
LastEditors: ZhangHongYu
LastEditTime: 2021-05-30 14:53:27
'''
'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-03-08 17:26:11
LastEditors: ZhangHongYu
LastEditTime: 2021-05-30 14:47:51
'''
import numpy as np
eps = 1e-6
n = 6
# 向量各元素之间互相依赖，不可并行/向量化
def SOR(A, b, w):
    assert(A.shape[0] == A.shape[1] == b.shape[0])
    x = np.zeros(b.shape, dtype=np.float32)
    # 迭代次数
    for t in range(n):
        for i in range(x.shape[0]):
            val = b[i]
            for j in range(A.shape[1]):
                if j != i :
                    val -= A[i, j] * x[j]
            x[i] = (1-w)*x[i] + w*(val/A[i][i])
        print(x)
    return x    

if __name__ == '__main__':
    # A一定要是主对角线占优矩阵
    A = np.array(
        [
            [3, 1, -1],
            [2, 4, 1],
            [-1, 2, 5]
        ],dtype=np.float32
    )
    b = np.array(
        [4, 1, 1],dtype=np.float32
    )
    w = 1.1
    x = SOR(A, b, w)
    print(x)
    #print(A, "\n", b)
    