import cv2
import matplotlib.pyplot as plt
# 读取图像
#img_path = 'assets/fish.jpg'

image = cv2.imread('assets/fish.jpg', cv2.IMREAD_GRAYSCALE)

# 进行Canny边缘检测
edges = cv2.Canny(image, 100, 200)  # 调整阈值参数以控制边缘检测的结果

# 保存边缘检测结果
#cv2.imwrite('fish_edges.png', edges)
plt.imsave('fish_edges.png', edges, cmap='binary')

# 显示原始图像和边缘检测结果
cv2.imshow('Original Image', image)
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()