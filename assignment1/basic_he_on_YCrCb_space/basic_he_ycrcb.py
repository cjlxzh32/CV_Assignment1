import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

# implementation of histogram equalization
def histogram_equalization(img_np):
    # 1. get the histogram
    hist, bins = np.histogram(img_np.flatten(), bins=256, range=[0,256])
    
    # 2. compute the probability distribution (PDF)
    pdf = hist / img_np.size
    
    # 3. compute the cumulative distribution (CDF)
    cdf = pdf.cumsum()
    
    # normalize to [0, 255]
    cdf_normalized = np.round(255 * cdf).astype(np.uint8)
    
    # 4. map pixels
    img_equalized = cdf_normalized[img_np]
    
    return img_equalized


input_dir = f"./input"
output_dir = f"./basic_he_on_YCrCb_space/output"
os.makedirs(output_dir, exist_ok=True)
# traverse all images in the directory
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        filepath = os.path.join(input_dir, filename)
        parts = os.path.splitext(filename)[0]
        save_path_result = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_result.png")
        save_path_combine = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_combine.png")
        save_path_pdf = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_result.pdf")
        if os.path.exists(save_path_combine) and os.path.exists(save_path_pdf):
            continue
        # read image
        img = cv2.imread(filepath)
        print(f"reading image: {filename}, size: {img.shape}")
        # histogram equalization
        # if color image
        if len(img.shape) == 3 and img.shape[2] == 3:
            # convert to YCrCb
            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            # luminance, chrominance-red, chrominance-blue
            y, cr, cb = cv2.split(ycrcb)
            # equalize the luminance channel
            y_eq = histogram_equalization(y)
            # merge channels
            ycrcb_eq = cv2.merge([y_eq, cr, cb])
            # convert back to BGR
            equalized_img_np = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)
        # if grayscale image
        elif len(img.shape) == 2:
            equalized_img_np = histogram_equalization(img)

        # visualization
        # only result
        plt.figure(figsize=(8,6))
        plt.imshow(equalized_img_np)
        plt.title("Equalized Image")
        plt.axis("off")   # 去掉坐标轴
        plt.savefig(save_path_result)
        plt.close()
        
        # combine result with origin
        plt.figure(figsize=(8,6))
        plt.subplot(1,2,1)
        plt.title("Original Image")
        plt.imshow(img)
        plt.axis("off")

        plt.subplot(1,2,2)
        plt.title("Equalized Image")
        plt.imshow(equalized_img_np)
        plt.axis("off")

        # plt.show()
        # save as png and PDF
        plt.savefig(save_path_combine)
        # plt.savefig(save_path_pdf, format='pdf')
        # plt.savefig("sample1_result.pdf", format='pdf', bbox_inches='tight', pad_inches=0)
        plt.close()
