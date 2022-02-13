'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-07-01 21:35:04
LastEditors: ZhangHongYu
LastEditTime: 2021-10-16 16:16:45
'''
import numpy as np
from sklearn.decomposition import PCA
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.sans-serif']=[u'SimHei']
mpl.rcParams['axes.unicode_minus']=False
# import matplotlib.font_manager as font_manager
# font_manager._rebuild()
def approximation(A, p):
    U, s, V_T = np.linalg.svd(A)
    B = np.zeros(A.shape)
    for i in range(p):
        B += s[i]*U[:,i].reshape(-1, 1).dot(V_T[i, :].reshape(1, -1))
    return B

if __name__ == '__main__':
    img = cv.imread("chapter12.特征值和奇异值/12.4.SVD的应用/12.4.3.图像压缩/img.jpeg", flags=0)
    img_output = img.copy()

    # p为近似矩阵的秩，秩p<=r，p越大图像压缩程度越小，越清晰
    p = 50
    img_output = approximation(img, p)
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(img)
    axs[0].set_title('原图')
    axs[1].imshow(img_output)
    axs[1].set_title('压缩后的图')
    plt.savefig('chapter12.特征值和奇异值/12.4.SVD的应用/12.4.3.图像压缩/result.png')
    plt.show()

