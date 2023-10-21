import matplotlib.pyplot as plt

# 示例数据
categories = ['A', 'B', 'C', 'D']  # 柱状图的类别标签
values1 = [20, 35, 30, 25]  # 第一个数据系列
values2 = [15, 25, 20, 35]  # 第二个数据系列
values3 = [10, 30, 25, 15]  # 第三个数据系列

# 绘制堆叠柱状图
plt.bar(categories, values1, label='Series 1')
plt.bar(categories, values2, bottom=values1, label='Series 2')
plt.bar(categories, values3, bottom=[values1[j] + values2[j] for j in range(len(values1))], label='Series 3')

# 添加图例
plt.legend()

# 显示图形
plt.show()