import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def pdf(x):
    return np.where(x <= 0.5, 12 * x ** 2, 12 * (1 - x) ** 2)


def cdf(a,steps=1000):
    x = np.linspace(0, 1, steps)
    pdf_val = pdf(x)
    cdf_val = np.cumsum(pdf_val)
    cdf_val = cdf_val / cdf_val[-1]
    return x, cdf_val


def he(img):
    #  Convert to HSI (approximately replaced by HSV)
    HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    H, S, V = cv2.split(HSV)
    I = V / 255.0

    hist, bins = np.histogram(I.flatten(), bins=256, range=(0, 1), density=True)
    cdf_in = np.cumsum(hist)
    cdf_in = cdf_in / cdf_in[-1]
    xs, cdf_target = cdf(np.linspace(0, 1, 256))
    mapping = np.interp(cdf_in, cdf_target, xs)

    I_eq = np.interp(I.flatten(), bins[:-1], mapping).reshape(I.shape)
    V_eq = np.clip(I_eq * 255, 0, 255).astype(np.uint8)
    HSV[:, :, 2] = V_eq
    out = cv2.cvtColor(HSV.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return out

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
            eq_img = he(img_rgb)
            eq_img_bgr = cv2.cvtColor(eq_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_img_path, eq_img_bgr)
            plot_his(img_rgb, eq_img, output_hist_path)
            print(f"{filename} done，save as {output_img_path}")

if __name__ == "__main__":
    input_folder = r"images"
    output_folder = r"01"
    process_folder(input_folder, output_folder)
    print("he done")

