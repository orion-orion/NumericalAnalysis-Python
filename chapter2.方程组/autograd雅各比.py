'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-06-19 18:59:48
LastEditors: ZhangHongYu
LastEditTime: 2021-06-19 19:01:21
'''
import torch
inp = torch.eye(5, requires_grad=True)
out = (inp+1).pow(2)
out.backward(torch.ones_like(inp), retain_graph=True)
print("First call\n", inp.grad)
out.backward(torch.ones_like(inp), retain_graph=True)
print("\nSecond call\n", inp.grad)
inp.grad.zero_()
out.backward(torch.ones_like(inp), retain_graph=True)
print("\nCall after zeroing gradients\n", inp.grad)