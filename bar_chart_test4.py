import matplotlib.pyplot as plt

# 示例数据
categories = ['A', 'B', 'C', 'D']  # 柱状图的类别标签
values = [20, 35, 30, 25]  # 每个类别对应的值

# 绘制柱状图
plt.bar(categories, values)

# 显示高度值
for i, value in enumerate(values):
    plt.text(i, value, str(value), ha='center', va='bottom')

# 显示图形
plt.show()