# This script batch segments pictures in <source> directory using SAM（Segment Anything）.
# And save the semantic pictures in <source_semantic> for colored versions,
# and <source_semanticB> for binary versions

# Author: Qihan Zhao

# Prerequisite:
# 1. download this script in directory `SAM`
# 2. download SAM weights in directory `SAM`
# 3. add <source> file in directory `SAM`

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import math
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

import argparse
parser = argparse.ArgumentParser("enlighten-anything")
parser.add_argument('--source_dir', type=str, default='data/LOL/our485/low', help='directory of data to be segmented')
args = parser.parse_args()




sam_checkpoint = "./segment_anything/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)


import os

sourcedir = args.source_dir
# read image 
filename = '719.png'
img_path = os.path.join(sourcedir, filename)
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# segment
masks = mask_generator.generate(image)

# save semantic
gray_img = np.zeros((masks[0]['segmentation'].shape[0], masks[0]['segmentation'].shape[1], 1), dtype=np.int8)
#img[:,:,3] = 0

original_gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

region_mean_list = []
for i, mask in enumerate(masks):
    mask_bool = mask['segmentation']

    region_mean = original_gray_image[mask_bool].mean()
    region_mean_list.append(region_mean)

sorted_indices = np.argsort(region_mean_list)

#逆序操作
#reverse_sorted_indices = sorted_indices[::-1]

masks_sorted = [masks[i] for i in sorted_indices]
region_mean_list_sorted = [region_mean_list[i] for i in sorted_indices]


#根据mask个数确定step
"""
step = 255.0 / len(masks)
for i, mask in enumerate(masks_sorted):
    mask_bool = mask['segmentation']    
    color_value = int(round((i+1) * step))
    gray_img[mask_bool] = color_value
cv2.imwrite(save_path, gray_img)
"""
step = 16
totoal_region_count = math.floor(256/step)

region_count = [0]*totoal_region_count

for i in range(len(region_mean_list_sorted)):
    group_index = math.floor(region_mean_list_sorted[i]/step)
    region_count[group_index] += 1

import matplotlib.pyplot as plt

# 示例数据
#categories = ['A', 'B', 'C', 'D']  # 柱状图的类别标签
#values = [20, 35, 30, 25]  # 每个类别对应的值

categories=[]
for i in range(totoal_region_count):
    categories.append(f'{i}')

# 创建画布和子图
fig, ax = plt.subplots()

# 绘制柱状图
plt.bar(categories, region_count)

# 设置背景色为浅灰色
#ax.set_facecolor('lightgray')

# 绘制柱状图

for i, value in enumerate(region_count):
    plt.text(i, value, str(value), ha='center', va='bottom')

# 添加标题和轴标签
#plt.title('Bar Chart')
plt.xlabel('Illumination Groups')
plt.ylabel('Region Counts')

# 设置图形大小
fig = plt.gcf()
fig.set_size_inches(6, 4)  # 设置为600x400像素

# 添加黑色边框
#plt.gca().spines['top'].set_visible(False)  # 隐藏顶部边框
#plt.gca().spines['right'].set_visible(False)  # 隐藏右侧边框
#plt.gca().spines['bottom'].set_color('black')  # 设置底部边框颜色为黑色
#plt.gca().spines['left'].set_color('black')  # 设置左侧边框颜色为黑色
# 设置背景色为浅灰色
fig.patch.set_facecolor('lightgray')
# 保存图像为RGB格式
plt.savefig('bar_chart_rgb.png', dpi=100, format='png')

# 使用Pillow库打开保存的图像
image = Image.open('bar_chart_rgb.png')

# 显示图形
plt.show()
