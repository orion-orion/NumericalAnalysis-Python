<!--
 * @Descripttion: 
 * @Version: 1.0
 * @Author: ZhangHongYu
 * @Date: 2021-09-19 19:53:53
 * @LastEditors: ZhangHongYu
 * @LastEditTime: 2022-07-02 19:13:10
-->
# 数值分析
[![Open Source Love](https://badges.frapsoft.com/os/v2/open-source.svg?v=103)](https://github.com/orion-orion/NumericalAnalysis)
[![](https://img.shields.io/github/license/orion-orion/Distributed-Algorithm-PySpark)](https://github.com/orion-orion/NumericalAnalysis/blob/master/LICENSE)
[![](https://img.shields.io/github/stars/orion-orion/NumericalAnalysis?style=social)](https://github.com/orion-orion/NumericalAnalysis)
## 1 简介
本项目为《数值分析》(Timothy Sauer著) 中的算法实现（使用Python+Numpy+Pytorch）。

## 2 目录
- 第0章 多项式求值
    - 霍纳法多项式求值
- 第1章 求解方程根
  - 不动点迭代 
  - 二分法
  - 牛顿法
- 第2章 方程组
  - 2.1 高斯消元 
    - 朴素高斯消元
  - 2.2 LU分解
    - LU分解及回代
  - 2.4 PA=LU分解
    - 部分主元法高斯消元
    - PA=LU分解及回代
  - 2.5 迭代方法
    - 三种迭代方法对比
    - 稀疏矩阵Jocobi迭代（没搞完）
    - Gauss-Seidel方法
    - Jocobi迭代
    - SOR方法
  - 2.6 用于对称正定矩阵的方法
    - 楚列斯基分解
    - 对角矩阵定义
    - 共轭梯度法
    - 预条件共轭梯度法 
  - 2.7 非线性方程组
    - 多变量牛顿方法
    - Broyden方法
    - Broyden方法2
- 第4章  最小二乘
  - 解析法求解最小二乘（直线拟合） [[算法讲解]](https://www.cnblogs.com/orion-orion/p/15887067.html) 
  - 解析法求解最小二乘（多项式拟合） [[算法讲解]](https://www.cnblogs.com/orion-orion/p/15887067.html) 
  - 范德蒙德矩阵
  - 范德蒙德矩阵实现最小二乘
  - 迭代法求解最小二乘 [[算法讲解]](https://www.cnblogs.com/orion-orion/p/15887067.html) 
  - 迭代法求解最小二乘（带正则项） [[算法讲解]](https://www.cnblogs.com/orion-orion/p/15887067.html) 
  - QR分解（经典Gram-Schmidt正交化）
  - QR分解实现最小二乘
  - 改进的Gram-Schmidt正交化
- 第9章 随机数和应用
  -  蒙特卡洛1型问题-随机数近似曲线下方面积
  -  蒙特卡洛2型问题-随机数近似图形面积
  -  随机游走
  -  最小标准生成器
- 第12章 特征值和特征向量
  - 12.1 幂迭代方法
    - 幂迭代法 [[算法讲解]](https://www.cnblogs.com/orion-orion/p/15405907.html)
    - 逆向幂迭代 [[算法讲解]](https://www.cnblogs.com/orion-orion/p/15405907.html)
    - 瑞利商迭代（结果与书上不符）  [[算法讲解]](https://www.cnblogs.com/orion-orion/p/15405907.html) 
    - 占优特征值 [[算法讲解]](https://www.cnblogs.com/orion-orion/p/15405907.html) 
  - 12.2 QR算法
    - 平移QR算法
    - 平移QR算法2
    - 无移动QR算法
    - PageRank算法 [[算法讲解]](https://www.cnblogs.com/orion-orion/p/15405907.html) 
    - QR算法
  - 12.3 奇异值分解
    -  奇异值分解
  - 12.4 奇异值分解的应用
    - 矩阵的低秩近似和降维 [[算法讲解]](https://www.cnblogs.com/orion-orion/p/15415610.html) 
    - 推荐系统应用 [[算法讲解]](https://www.cnblogs.com/orion-orion/p/15415610.html) 
    - 图像压缩 [[算法讲解]](https://www.cnblogs.com/orion-orion/p/15415610.html) 
- 第13章 最优化
  -  不使用导数的无约束优化
     -  黄金分割搜索 [[算法讲解]](https://www.cnblogs.com/orion-orion/p/15418056.html) 
  -  使用导数的无约束优化
     -  共轭梯度法 [[算法讲解]](https://www.cnblogs.com/orion-orion/p/15418056.html) 
     -  牛顿法 [[算法讲解]](https://www.cnblogs.com/orion-orion/p/15418056.html) 
     -  最速下降法 [[算法讲解]](https://www.cnblogs.com/orion-orion/p/15418056.html) 

