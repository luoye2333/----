import matplotlib.pyplot as plt
# 输入数据
x1 = [1, 2, 3, 4, 5]
y1 = [2, 5, 3, 8, 6]
x2 = [1, 2, 3, 4, 5]
y2 = [5, 1, 7, 4, 9]

# 创建两个Figure对象
fig1 = plt.figure(1)
fig2 = plt.figure(2)

# 在每个Figure对象中创建Axes对象
ax1 = fig1.add_subplot(1, 1, 1)
ax2 = fig2.add_subplot(1, 1, 1)

# 在每个Axes对象中绘制图像
ax1.plot(x1, y1)
ax2.plot(x2, y2)

# 显示图像
plt.show()