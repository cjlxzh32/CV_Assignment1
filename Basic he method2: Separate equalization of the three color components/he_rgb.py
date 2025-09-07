import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def equalize(channel):
    hist = np.zeros(256, dtype=np.int32)
    for value in channel.flatten():
        hist[value] += 1
    num = channel.size
    pdf = hist / num
    cdf = np.cumsum(pdf)
    a = np.floor(255 * cdf).astype(np.uint8)
    equalized = a[channel]
    return equalized

def he_rgb(img):
    r, g, b = cv2.split(img)
    r_eq = equalize(r)
    g_eq = equalize(g)
    b_eq = equalize(b)
    eq_img = cv2.merge((r_eq, g_eq, b_eq))
    return eq_img

def plot_his(img_before, img_after, save_path):
    colors = ['r', 'g', 'b']
    labels = ['R channel', 'G channel', 'B channel']
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for i, color in enumerate(colors):
        plt.hist(img_before[:, :, i].flatten(), bins=256, range=[0, 256],
                 color=color, alpha=0.5, label=labels[i])
    plt.title("Before Equalization")
    plt.xlabel("pixel value")
    plt.ylabel("pixel number")
    plt.legend()
    plt.subplot(1, 2, 2)
    for i, color in enumerate(colors):
        plt.hist(img_after[:, :, i].flatten(), bins=256, range=[0, 256],
                 color=color, alpha=0.5, label=labels[i])
    plt.title("After Equalization")
    plt.xlabel("pixel value")
    plt.ylabel("pixel number")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def process_folder(inputf, outputf):
    if not os.path.exists(outputf):
        os.makedirs(outputf)
    for filename in os.listdir(inputf):
        if filename.lower().endswith(('.jpg', '.jpeg')):
            input_path = os.path.join(inputf, filename)
            output_img_path = os.path.join(outputf, f"eq_{filename}")
            output_hist_path = os.path.join(outputf, f"eq_{os.path.splitext(filename)[0]}_hist.png")
            img = cv2.imread(input_path)
            if img is None:
                print(f"cannot read {input_path}")
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            eq_img = he_rgb(img_rgb)
            eq_img_bgr = cv2.cvtColor(eq_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_img_path, eq_img_bgr)
            plot_his(img_rgb, eq_img, output_hist_path)
            print(f"{filename} done，save as {output_img_path}")

if __name__ == "__main__":
    input_folder = r"images"
    output_folder = r"results"
    process_folder(input_folder, output_folder)
    print("he done")
