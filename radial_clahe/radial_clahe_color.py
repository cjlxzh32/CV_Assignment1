import cv2
import numpy as np
import os

def soft_radial_clahe_color(img, num_rings=100, clipLimit=2.0, nbins=256, sigma_r=5.0):
    """
    对彩色图像每个通道应用全局 Soft Radial CLAHE
    """
    h, w, c = img.shape
    output = np.zeros_like(img, dtype=np.uint8)
    center = (h/2, w/2)
    max_radius = np.sqrt(center[0]**2 + center[1]**2)

    # 计算每个像素到中心距离
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((Y - center[0])**2 + (X - center[1])**2)
    normalized_dist = dist / max_radius * num_rings

    # 构建全局高斯权重矩阵 shape: (num_rings, h, w)
    ring_indices = np.arange(num_rings).reshape(-1,1,1)
    normalized_dist_3d = normalized_dist[np.newaxis, :, :]
    weights = np.exp(-0.5 * ((ring_indices - normalized_dist_3d)/sigma_r)**2)
    weights /= weights.sum(axis=0, keepdims=True)

    # 对每个通道单独处理
    for ch in range(c):
        channel = img[:,:,ch]
        # 计算每个环的映射表
        mappings = np.zeros((num_rings, nbins), dtype=np.float32)
        for i in range(num_rings):
            r1 = i * max_radius / num_rings
            r2 = (i + 1) * max_radius / num_rings
            mask = (dist >= r1) & (dist < r2)
            pixels = channel[mask]
            if len(pixels) > 0:
                hist, _ = np.histogram(pixels, bins=nbins, range=(0,256))
                clip_val = clipLimit * hist.mean()
                excess = hist - clip_val
                excess[excess < 0] = 0
                hist = np.minimum(hist, clip_val)
                hist += excess.sum() // nbins
                cdf = np.cumsum(hist) / np.sum(hist)
                mappings[i] = cdf * 255
            else:
                mappings[i] = np.arange(nbins, dtype=np.float32)

        # 输出初始化
        out_channel = np.zeros_like(channel, dtype=np.float32)
        for val in range(256):
            mask = (channel == val)
            if np.any(mask):
                weighted_map = np.sum(weights[:, mask] * mappings[:, val][:, np.newaxis], axis=0)
                out_channel[mask] = weighted_map

        output[:,:,ch] = np.clip(out_channel, 0, 255).astype(np.uint8)

    return output


def process_folder_color(input_folder, output_folder, num_rings=100, clipLimit=2.0, sigma_r=5.0):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".jpg", ".jpeg")):
            filepath = os.path.join(input_folder, filename)
            img = cv2.imread(filepath)
            if img is None:
                print(f"Skipped unreadable file: {filename}")
                continue

            processed = soft_radial_clahe_color(img, num_rings=num_rings, clipLimit=clipLimit, sigma_r=sigma_r)

            save_path = os.path.join(output_folder, filename)
            cv2.imwrite(save_path, processed)
            print(f"Processed: {filename} → {save_path}")


if __name__ == "__main__":
    input_folder = "input"
    output_folder = "output_color"
    process_folder_color(input_folder, output_folder, num_rings=100, clipLimit=2.0, sigma_r=5.0)
