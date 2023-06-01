import matplotlib.pyplot as plt

# 设置点的坐标和路径（假设已经有了解决方案的路径）
x = [50, 20, 70, 10, 60, 30, 80, 40, 90, 50]
y = [50, 80, 80, 20, 30, 60, 50, 90, 30, 20]
sample_path = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
optimal_path = [0, 4, 5, 1, 7, 6, 3, 9, 8, 2, 0]

# 绘制点
plt.scatter(x, y, color="blue")

# 绘制样例路径
for i in range(len(sample_path)-1):
    start_x, start_y = x[sample_path[i]], y[sample_path[i]]
    end_x, end_y = x[sample_path[i+1]], y[sample_path[i+1]]
    plt.plot([start_x, end_x], [start_y, end_y], color="gray", linewidth=1)
    plt.draw()
    plt.pause(0.5)
    
# 绘制最优路径
for i in range(len(optimal_path)-1):
    start_x, start_y = x[optimal_path[i]], y[optimal_path[i]]
    end_x, end_y = x[optimal_path[i+1]], y[optimal_path[i+1]]
    plt.plot([start_x, end_x], [start_y, end_y], color="red", linewidth=2)

    # 更新图形并暂停
    plt.draw()
    plt.pause(0.5)

# 显示最终路径
plt.plot([x[optimal_path[-1]], x[optimal_path[0]]], [y[optimal_path[-1]], y[optimal_path[0]]], color="red", linewidth=2)
plt.draw()

# 防止绘图窗口闪退
while True:
    try:
        plt.pause(0.1)
    except:
        break