import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams["font.sans-serif"] = [u"SimHei"]
mpl.rcParams["axes.unicode_minus"] = False


def approximation(A, p):
    B = np.zeros(A.shape)
    for c in range(A.shape[2]):
        U, s, V_T = np.linalg.svd(A[:, :, c])
        for i in range(p):
            B[:, :, c] += s[i] * \
                U[:, i].reshape(-1, 1).dot(V_T[i, :].reshape(1, -1))
    return B


if __name__ == '__main__':
    img = cv.imread(
        "chapter12.特征值和奇异值/12.4.SVD的应用/12.4.3.图像压缩/img.jpeg")
    # 将OpenCV采用的BGR格式转换到Matplotlib采用的RGB格式
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # 图像必须归一化到[0 - 1]范围
    img = img.astype(np.float32) / 255.0
    img_output = img.copy()

    # p为近似矩阵的秩，秩p<=r，p越大图像压缩程度越小，越清晰
    p = 50
    img_output = approximation(img, p)

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(np.clip(img, 0, 1))
    axs[0].set_title(u"原图")
    axs[1].imshow(np.clip(img_output, 0, 1))
    axs[1].set_title(u"压缩后的图")
    plt.savefig(
        "chapter12.特征值和奇异值/12.4.SVD的应用/12.4.3.图像压缩/result.png",
        bbox_inches="tight")
    plt.show()
