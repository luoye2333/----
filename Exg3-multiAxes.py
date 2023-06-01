import matplotlib.pyplot as plt

# 输入数据
x1 = [1, 2, 3, 4, 5]
y1 = [2, 5, 3, 8, 6]
x2 = [1, 2, 3, 4, 5]
y2 = [5, 1, 7, 4, 9]

# 创建两个子图
fig, (ax1, ax2) = plt.subplots(1, 2)

# 在子图中绘制图像
ax1.plot(x1, y1)
ax2.plot(x2, y2)

# 显示图像
plt.show()