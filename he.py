import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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
    
    return img_equalized, hist, cdf_normalized


input_dir = f"input"
output_dir = f"output"
os.makedirs(output_dir, exist_ok=True)
# traverse all images in the directory
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        filepath = os.path.join(input_dir, filename)
        parts = os.path.splitext(filename)[0]
        save_path_png = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_result.png")
        save_path_pdf = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_result.pdf")
        if os.path.exists(save_path_png) and os.path.exists(save_path_pdf):
            continue
        # read a grayscale image
        img = Image.open(filepath).convert("L")
        print(f"reading image: {filename}, size: {img.size}")
        # convert to numpy.ndarray for for easier processing later 
        img_np = np.array(img)
        # histogram equalization
        equalized_img_np, hist, cdf_norm = histogram_equalization(img_np)
        # equalized_img = Image.fromarray(equalized_img_np)

        # visualization
        plt.figure(figsize=(8,6))

        plt.subplot(1,2,1)
        plt.title("Original Image")
        plt.imshow(img, cmap='gray')
        plt.axis("off")

        plt.subplot(1,2,2)
        plt.title("Equalized Image")
        plt.imshow(equalized_img_np, cmap='gray')
        plt.axis("off")

        # plt.show()
        # save as png and PDF
        plt.savefig(save_path_png)
        plt.savefig(save_path_pdf, format='pdf')
        # plt.savefig("sample1_result.pdf", format='pdf', bbox_inches='tight', pad_inches=0)