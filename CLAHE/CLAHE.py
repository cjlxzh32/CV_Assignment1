import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def clip_his(hist, clip_limit):
    excess = hist - clip_limit
    excess[excess < 0] = 0
    n_excess = excess.sum()
    hist = np.minimum(hist, clip_limit)
    hist += n_excess / hist.size
    return hist


def clahe(img, n_bins=256, clip_limit=0.01, tile_grid_size=(8, 8)):
    h, w, _ = img.shape
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    L, a, b = cv2.split(lab)
    t_h = h // tile_grid_size[0]
    t_w = w // tile_grid_size[1]
    clip_val = clip_limit * t_h * t_w

    tile_maps = [[None for _ in range(tile_grid_size[1])] for _ in range(tile_grid_size[0])]
    for i in range(tile_grid_size[0]):
        for j in range(tile_grid_size[1]):
            r0 = i * t_h
            r1 = (i + 1) * t_h if i < tile_grid_size[0] - 1 else h
            c0 = j * t_w
            c1 = (j + 1) * t_w if j < tile_grid_size[1] - 1 else w
            tile = L[r0:r1, c0:c1]
            hist, bins = np.histogram(tile.flatten(), n_bins, [0, n_bins])
            hist = clip_his(hist, clip_val)
            cdf = hist.cumsum()
            cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min()) * (n_bins - 1)
            lut = np.interp(np.arange(256), bins[:-1], cdf)
            tile_maps[i][j] = lut

    # bilinear interpolation
    L_eq = np.zeros_like(L, dtype=np.float32)
    for y in range(h):
        for x in range(w):
            gx = x / t_w - 0.5
            gy = y / t_h - 0.5
            i = int(np.floor(gy))
            j = int(np.floor(gx))
            i0 = max(min(i, tile_grid_size[0] - 1), 0)
            j0 = max(min(j, tile_grid_size[1] - 1), 0)
            i1 = min(i0 + 1, tile_grid_size[0] - 1)
            j1 = min(j0 + 1, tile_grid_size[1] - 1)
            dy = gy - i
            dx = gx - j
            val = L[y, x]
            v00 = tile_maps[i0][j0][val]
            v01 = tile_maps[i0][j1][val]
            v10 = tile_maps[i1][j0][val]
            v11 = tile_maps[i1][j1][val]
            L_eq[y, x] = (
                (1 - dx) * (1 - dy) * v00 +
                dx * (1 - dy) * v01 +
                (1 - dx) * dy * v10 +
                dx * dy * v11
            )

    L_eq = np.clip(L_eq, 0, 255).astype(np.uint8)
    lab_eq = cv2.merge([L_eq, a, b])
    img_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
    return img_eq

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
            eq_img = clahe(img_rgb)
            eq_img_bgr = cv2.cvtColor(eq_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_img_path, eq_img_bgr)
            plot_his(img_rgb, eq_img, output_hist_path)
            print(f"{filename} done，save as {output_img_path}")

if __name__ == "__main__":
    input_folder = r"images"  # 输入文件夹路径
    output_folder = r"clahe_results"  # 输出文件夹路径
    process_folder(input_folder, output_folder)
    print("HE done")