import matplotlib.pyplot as plt
import numpy as np

# 创建数据
x = [16, 32, 48, 64]
y1 = [16,34,44,68]
y2 = []
y3 = np.random.rand(len(x)) * 100
y4 = np.random.rand(len(x)) * 100

# 设置柱状图的位置和宽度
bar_width = 0.2
index = np.arange(len(x))

# 创建一个图形对象和子图
fig, ax = plt.subplots()

# 绘制四组条形图
ax.bar(index, y1, bar_width, label='Group 1', color='b')
ax.bar(index + bar_width, y2, bar_width, label='Group 2', color='r')
ax.bar(index + 2 * bar_width, y3, bar_width, label='Group 3', color='g')
ax.bar(index + 3 * bar_width, y4, bar_width, label='Group 4', color='m')

# 添加标题和标签
ax.set_title('Grouped Bar Chart Example', fontsize=16)
ax.set_xlabel('X Axis Label', fontsize=14)
ax.set_ylabel('Y Axis Label', fontsize=14)
ax.set_xticks(index + 1.5 * bar_width)
ax.set_xticklabels(x, fontsize=12)

# 添加网格
ax.grid(True, linestyle='--', alpha=0.7)

# 添加图例
ax.legend(loc='upper left', fontsize=12)

# 调整布局，以确保不被剪切
plt.tight_layout()

# 保存图形为图片
plt.savefig('grouped_bar_chart.png', dpi=300)

# 显示图形
plt.show()
