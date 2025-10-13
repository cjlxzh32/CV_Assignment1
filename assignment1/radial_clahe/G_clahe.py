import cv2
import numpy as np
import os

def radial_soft_clahe_luminance(y_channel, num_rings=100, clipLimit=2.0, nbins=256, sigma_r=5.0, alpha=0.5):

    h, w = y_channel.shape
    center = (h/2, w/2)
    max_radius = np.sqrt(center[0]**2 + center[1]**2)

    hist, _ = np.histogram(y_channel.flatten(), bins=nbins, range=(0,256))
    cdf = hist.cumsum()
    cdf_normalized = cdf * 255 / cdf[-1]
    he_img = cdf_normalized[y_channel]

    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((Y - center[0])**2 + (X - center[1])**2)
    normalized_dist = dist / max_radius * num_rings

    mappings = np.zeros((num_rings, nbins), dtype=np.float32)
    for i in range(num_rings):
        r1 = i * max_radius / num_rings
        r2 = (i + 1) * max_radius / num_rings
        mask = (dist >= r1) & (dist < r2)
        pixels = y_channel[mask]
        if len(pixels) > 0:
            hist_local, _ = np.histogram(pixels, bins=nbins, range=(0,256))
            clip_val = clipLimit * hist_local.mean()
            excess = hist_local - clip_val
            excess[excess < 0] = 0
            hist_local = np.minimum(hist_local, clip_val)
            hist_local += excess.sum() // nbins
            cdf_local = np.cumsum(hist_local) / np.sum(hist_local)
            mappings[i] = cdf_local * 255
        else:
            mappings[i] = np.arange(nbins, dtype=np.float32)

    ring_indices = np.arange(num_rings).reshape(-1,1,1)
    normalized_dist_3d = normalized_dist[np.newaxis, :, :]
    weights = np.exp(-0.5 * ((ring_indices - normalized_dist_3d)/sigma_r)**2)
    weights /= weights.sum(axis=0, keepdims=True)

    out_y = np.zeros_like(y_channel, dtype=np.float32)
    for val in range(256):
        mask = (y_channel == val)
        if np.any(mask):
            weighted_map = np.sum(weights[:, mask] * mappings[:, val][:, np.newaxis], axis=0)
            out_y[mask] = weighted_map

    fused = alpha * out_y + (1-alpha) * he_img
    return np.clip(fused, 0, 255).astype(np.uint8)


def process_folder_luminance(input_folder, output_folder, num_rings=100, clipLimit=2.0, sigma_r=5.0, alpha=0.5):
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

            y_clahe = radial_soft_clahe_luminance(y, num_rings=num_rings,
                                                  clipLimit=clipLimit,
                                                  sigma_r=sigma_r,
                                                  alpha=alpha)

            ycrcb_clahe = cv2.merge([y_clahe, cr, cb])
            processed = cv2.cvtColor(ycrcb_clahe, cv2.COLOR_YCrCb2BGR)

            save_path = os.path.join(output_folder, filename)
            cv2.imwrite(save_path, processed)
            print(f"Processed: {filename} → {save_path}")


if __name__ == "__main__":
    input_folder = "input"
    output_folder = "output_G_clahe"
    process_folder_luminance(input_folder, output_folder, num_rings=100, clipLimit=2.0, sigma_r=5.0, alpha=0.5)
