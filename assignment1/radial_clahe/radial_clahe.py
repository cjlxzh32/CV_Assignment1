import cv2
import numpy as np
import os

def radial_clahe_cartesian(gray_img, num_rings=30, clipLimit=2.0, nbins=256, blend_width=40):

    h, w = gray_img.shape
    center = (h / 2, w / 2)
    max_radius = np.sqrt(center[0]**2 + center[1]**2)

    radius_step = max_radius / num_rings

    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((Y - center[0])**2 + (X - center[1])**2)

    mappings = []
    for i in range(num_rings):
        r1 = i * radius_step
        r2 = (i + 1) * radius_step
        mask = (dist >= r1) & (dist < r2)
        ring_pixels = gray_img[mask]
        if len(ring_pixels) > 0:
            hist, _ = np.histogram(ring_pixels, bins=nbins, range=(0,256))
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
    for i in range(num_rings):
        r1 = i * radius_step
        r2 = (i + 1) * radius_step
        mask = (dist >= r1) & (dist < r2)
        pixels = gray_img[mask]
        vals = mappings[i][pixels]

        # Smooth transition using S-curve
        if i < num_rings - 1:
            blend_mask = (dist[mask] > (r2 - blend_width))
            alpha = (dist[mask][blend_mask] - (r2 - blend_width)) / blend_width
            alpha = alpha**2 * (3 - 2*alpha)
            vals[blend_mask] = (1 - alpha) * mappings[i][pixels[blend_mask]] + \
                               alpha * mappings[i+1][pixels[blend_mask]]

        output[mask] = vals

    return np.clip(output, 0, 255).astype('uint8')


def process_folder(input_folder, output_folder, num_rings=30, clipLimit=2.0, blend_width=40):

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

            y_clahe = radial_clahe_cartesian(y, num_rings=num_rings, clipLimit=clipLimit, blend_width=blend_width)

            ycrcb_clahe = cv2.merge([y_clahe, cr, cb])
            processed = cv2.cvtColor(ycrcb_clahe, cv2.COLOR_YCrCb2BGR)

            save_path = os.path.join(output_folder, filename)
            cv2.imwrite(save_path, processed)
            print(f"Processed: {filename} → {save_path}")


if __name__ == "__main__":
    input_folder = "input"
    output_folder = "output"
    process_folder(input_folder, output_folder, num_rings=30, clipLimit=2.0, blend_width=40)
