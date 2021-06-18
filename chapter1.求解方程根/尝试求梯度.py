'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-03-08 16:41:50
LastEditors: ZhangHongYu
LastEditTime: 2021-06-06 20:51:56
'''
import torch
#这里我们对w,b求梯度，事实上只要requires_grad=True,我们也可以对x,y求梯度
x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output  
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
# print(z)
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
print(loss)  # loss是一个标量tensor(3.2646)，只有这样才可反向传播
loss.backward()
print(w.grad,'\n', b.grad)
print(w.grad!=None)