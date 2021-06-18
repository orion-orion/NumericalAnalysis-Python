'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-06-06 21:48:48
LastEditors: ZhangHongYu
LastEditTime: 2021-06-06 21:58:20
'''
import torch
a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)
Q = 3*a**3 - b**2
external_grad = torch.tensor([1., 1.])
Q.backward(gradient=external_grad)
a.grad.zero_()
b.grad.zero_()
with torch.no_grad():
    a += 1
Q = 5*a*b
Q.backward(gradient=external_grad)
print(a.grad)
print(b.grad)