import matplotlib.pyplot as plt
import numpy as np

# 示例数据
categories = ['A', 'B', 'C', 'D']  # 柱状图的类别标签
segments = [10, 3, 5, 2]  # 每个类别对应的段数

# 生成不同颜色的颜色映射
cmap = plt.get_cmap('tab10')  # 使用tab10颜色映射
colors = cmap(np.linspace(0, 1, sum(segments)))  # 根据总段数生成颜色数组

# 绘制柱状图并设置每一段的颜色
start = 0
for i, segment in enumerate(segments):
    end = start + segment
    plt.bar(categories[i], segment, color=colors[start:end])
    start = end

# 显示图形
plt.show()