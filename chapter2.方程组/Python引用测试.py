'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-05-29 16:54:04
LastEditors: ZhangHongYu
LastEditTime: 2021-05-29 16:56:55
'''
# 如果是对象，则始终为引用
matrix = [
            [1, 2, 3],
            [4, 5, 6]
        ]
refer = matrix
refer[0][1] = 9
print(matrix)

# 如果是内置类型，刚开始是引用，一旦修改b，则变为拷贝
a = 6
print(id(a))
b = a
print(id(b))
b = 7
print(id(b))
print(a)