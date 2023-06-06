import numpy as np

def softmax(x):
    x-=np.max(x)#防止溢出
    return np.exp(x) / np.sum(np.exp(x), axis=0)

# 示例数组
x = np.array([1, 2, 3, 4, 5])

# 对数组使用softmax函数
softmax_x = softmax(x)

print(softmax_x)


# x-=np.max(x)#防止溢出
# softmax_x=np.exp(x) / np.sum(np.exp(x), axis=0)
# print(softmax_x)

x-=np.max(x)#防止溢出
softmax_x=np.exp(x) / np.sum(np.exp(x))
print(softmax_x)