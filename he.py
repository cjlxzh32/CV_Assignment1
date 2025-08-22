import numpy as np
import matplotlib.pyplot as plt
import cv2  # 这里只用来读取图像和显示，可以换成 PIL

def histogram_equalization(img):
    # 1. 统计直方图
    hist, bins = np.histogram(img.flatten(), bins=256, range=[0,256])
    
    # 2. 计算概率分布 (PDF)
    pdf = hist / img.size
    
    # 3. 计算累计分布 (CDF)
    cdf = pdf.cumsum()
    
    # 归一化到 [0,255]
    cdf_normalized = np.round(255 * cdf).astype(np.uint8)
    
    # 4. 像素映射
    img_equalized = cdf_normalized[img]
    
    return img_equalized, hist, cdf_normalized

# 读取灰度图像
img = cv2.imread("sample.jpg", cv2.IMREAD_GRAYSCALE)

# 直方图均衡化
equalized_img, hist, cdf_norm = histogram_equalization(img)

# 可视化对比
plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(img, cmap='gray')
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Equalized Image (NumPy)")
plt.imshow(equalized_img, cmap='gray')
plt.axis("off")

plt.show()
