import matplotlib.pyplot as plt
import numpy as np


# 创建数据
x = [16, 32, 48, 64]

y1 = [16,50,94,162]
y2 = [15,38,77,147]
y3 = [13,46,82,139]
y4 = [12,35,66,121]
'''
y1 = [16,34,44,68]
y2 = [15,23,39,70]
y3 = [13,33,36,57]
y4 = [12,23,31,55]
'''
# 创建一个图形对象和子图
fig, ax = plt.subplots()

# 绘制四条折线
ax.plot(x, y1, marker='o', linestyle='-', color='b', label='only Final Expert')#Final Decision Expert
ax.plot(x, y2, marker='s', linestyle='--', color='r', label='with SO Expert')#Second Opinion Expert
ax.plot(x, y3, marker='^', linestyle='-.', color='g', label='with PR Expert')#Prompt Retention Expert
ax.plot(x, y4, marker='d', linestyle=':', color='m', label='MoLE')

# 添加标题和标签
#ax.set_title('Multi-Line Plot Example', fontsize=16)
ax.set_xlabel('Generated Tokens', fontsize=14)
ax.set_ylabel('Number of Hallucinations', fontsize=14)

# 添加网格
ax.grid(True, linestyle='--', alpha=0.7)

# 设置x轴和y轴的刻度字体大小
ax.tick_params(axis='both', which='major', labelsize=12)

# 添加图例
ax.legend(loc='upper left', fontsize=12)

# 调整布局，以确保不被剪切
plt.tight_layout()

# 保存图形为图片
plt.savefig('multi_line_plot.png', dpi=300)