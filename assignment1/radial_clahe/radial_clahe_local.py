import cv2
import numpy as np
import os

def soft_radial_clahe(gray_img, num_rings=60, clipLimit=2.0, nbins=256, sigma_r=1.0, neighbor_rings=2):

    h, w = gray_img.shape
    center = (h/2, w/2)
    max_radius = np.sqrt(center[0]**2 + center[1]**2)

    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((Y - center[0])**2 + (X - center[1])**2)
    normalized_dist = dist / max_radius * num_rings  # 虚拟浮点环索引

    mappings = []
    for i in range(num_rings):
        r1 = i * max_radius / num_rings
        r2 = (i + 1) * max_radius / num_rings
        mask = (dist >= r1) & (dist < r2)
        pixels = gray_img[mask]
        if len(pixels) > 0:
            hist, _ = np.histogram(pixels, bins=nbins, range=(0,256))
            clip_val = clipLimit * hist.mean()
            excess = hist - clip_val
            excess[excess < 0] = 0
            hist = np.minimum(hist, clip_val)
            hist += excess.sum() // nbins
            cdf = np.cumsum(hist) / np.sum(hist)
            mapping = np.round(cdf * 255).astype('uint8')
        else:
            mapping = np.arange(256, dtype='uint8')
        mappings.append(mapping)

    output = np.zeros_like(gray_img, dtype=np.float32)
    for i in range(h):
        for j in range(w):
            d = normalized_dist[i,j]
            idx_center = int(np.floor(d))
            start = max(idx_center - neighbor_rings, 0)
            end = min(idx_center + neighbor_rings + 1, num_rings)
            ring_indices = np.arange(start, end)

            weights = np.exp(-0.5 * (ring_indices - d)**2 / sigma_r**2)
            weights /= weights.sum()
            val = gray_img[i,j]

            mapped_val = 0.0
            for w_idx, r_idx in zip(weights, ring_indices):
                mapped_val += w_idx * mappings[r_idx][val]
            output[i,j] = mapped_val

    return np.clip(output, 0, 255).astype('uint8')


def process_folder(input_folder, output_folder, num_rings=60, clipLimit=2.0):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".jpg", ".jpeg")):
            filepath = os.path.join(input_folder, filename)
            img = cv2.imread(filepath)
            if img is None:
                print(f"Skipped unreadable file: {filename}")
                continue

            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(ycrcb)

            y_clahe = soft_radial_clahe(y, num_rings=num_rings, clipLimit=clipLimit, sigma_r=1.0, neighbor_rings=2)

            ycrcb_clahe = cv2.merge([y_clahe, cr, cb])
            processed = cv2.cvtColor(ycrcb_clahe, cv2.COLOR_YCrCb2BGR)

            save_path = os.path.join(output_folder, filename)
            cv2.imwrite(save_path, processed)
            print(f"Processed: {filename} → {save_path}")


if __name__ == "__main__":
    input_folder = "input"
    output_folder = "output_local"
    process_folder(input_folder, output_folder, num_rings=80, clipLimit=2.0)

