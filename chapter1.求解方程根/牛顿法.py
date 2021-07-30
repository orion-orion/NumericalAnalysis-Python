'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-03-08 15:57:56
LastEditors: ZhangHongYu
LastEditTime: 2021-07-24 15:26:54
'''
import numpy as np
import math
import torch
#x.grad为dy/dx(假设dy为最后一个节点)
def newton(x0, k, f): #迭代k次,包括x0在内共k+1个数
    # 初始化计算图参数
    x = torch.tensor([x0], requires_grad=True)
    for i in range(1, k+1):
        # 每次迭代都要调用函数前向传播一次并重新求梯度，类似于网络训练，然后更新变量(这里是x)
        # 前向传播，注意x要用新的对象，否则后面y.backgrad后会释放
        y = f(x)
        # 因为调用backward的对象自身不算梯度，相当于dy/dy = 1-> y.grad其实还是None
        # 或者令loss=y.sum() loss.backward()
        # y.backward(gradient = torch.tensor([1.0]))
        y.backward() # y.grad是None
        # 更新参数
        # 注意此处必须调用x.add原地修改，否则会生成一份copy，调用.grad得None
        # 修改式需关闭求导，否则会对leaf变量就地修改未定义行为(修改是计算图的一部分?)
        with torch.no_grad(): 
            x.sub_(torch.divide(y, x.grad))   
        x.grad.zero_() # 清空梯度，使下一轮建立新的计算图，否则因为backward释放资源下一轮再backward出错
        #注意x.grad不能是0，否则要出错使g(x)/x.grad变为none
        # 这里g(x)为cos(x)就不行，这样在x=0 时, -sin0=0，故x.grad = 0
    return x.detach().numpy()[0]
if __name__ == '__main__':
    f = lambda x: x**3 + x - 1
    x0 = 1.0
    res = newton(x0, 10, f)
    print(res)