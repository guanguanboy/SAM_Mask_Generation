import matplotlib.pyplot as plt

# 示例数据
categories = ['A', 'B', 'C', 'D']  # 柱状图的类别标签
values = [20, 35, 30, 25]  # 每个类别对应的值

# 绘制柱状图
plt.bar(categories, values)

# 添加标题和轴标签
#plt.title('Bar Chart')
plt.xlabel('Illumination Groups')
plt.ylabel('Region Counts')

# 显示图形
plt.show()