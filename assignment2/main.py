import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import time
from disparity_computation import *

# (name, function, optional per-method kwargs)
METHODS = [
    # Baseline methods
    ("ssd",      lambda l, r, **kw: compute_disparity(l, r, method="SSD", **kw), {"block_size": 16}),
    ("ssd+sobel",    lambda l, r, **kw: compute_disparity_enhanced(l, r, method="SSD", **kw), {"block_size": 16, "use_sobel": True, "use_consistency_check": False}),
    ("ssd+consistency",    lambda l, r, **kw: compute_disparity_enhanced(l, r, method="SSD", **kw), {"block_size": 16, "use_sobel": False, "use_consistency_check": True}),
    ("ssd+sobel+consistency",    lambda l, r, **kw: compute_disparity_enhanced(l, r, method="SSD", **kw), {"block_size": 16, "use_sobel": True, "use_consistency_check": True}),

    ("sad",      lambda l, r, **kw: compute_disparity(l, r, method="SAD", **kw), {"block_size": 16}),
    ("sad+sobel",    lambda l, r, **kw: compute_disparity_enhanced(l, r, method="SAD", **kw), {"block_size": 16, "use_sobel": True, "use_consistency_check": False}),
    ("sad+consistency",    lambda l, r, **kw: compute_disparity_enhanced(l, r, method="SAD", **kw), {"block_size": 16, "use_sobel": False, "use_consistency_check": True}),
    ("sad+sobel+consistency",    lambda l, r, **kw: compute_disparity_enhanced(l, r, method="SAD", **kw), {"block_size": 16, "use_sobel": True, "use_consistency_check": True}),

    ("zsad",      lambda l, r, **kw: compute_disparity(l, r, method="ZSAD", **kw), {"block_size": 16}),
    ("zsad+sobel",    lambda l, r, **kw: compute_disparity_enhanced(l, r, method="ZSAD", **kw), {"block_size": 16, "use_sobel": True, "use_consistency_check": False}),
    ("zsad+consistency",    lambda l, r, **kw: compute_disparity_enhanced(l, r, method="ZSAD", **kw), {"block_size": 16, "use_sobel": False, "use_consistency_check": True}),
    ("zsad+sobel+consistency",    lambda l, r, **kw: compute_disparity_enhanced(l, r, method="ZSAD", **kw), {"block_size": 16, "use_sobel": True, "use_consistency_check": True}),

    ("zncc",      lambda l, r, **kw: compute_disparity(l, r, method="ZNCC", **kw), {"block_size": 16}),
    ("zncc+sobel",    lambda l, r, **kw: compute_disparity_enhanced(l, r, method="ZNCC", **kw), {"block_size": 16, "use_sobel": True, "use_consistency_check": False}),
    ("zncc+consistency",    lambda l, r, **kw: compute_disparity_enhanced(l, r, method="ZNCC", **kw), {"block_size": 16, "use_sobel": False, "use_consistency_check": True}),
    ("zncc+sobel+consistency",    lambda l, r, **kw: compute_disparity_enhanced(l, r, method="ZNCC", **kw), {"block_size": 16, "use_sobel": True, "use_consistency_check": True}),

    ("census",      lambda l, r, **kw: compute_disparity(l, r, method="CENSUS", **kw), {"block_size": 16}),
    ("census+sobel",    lambda l, r, **kw: compute_disparity_enhanced(l, r, method="CENSUS", **kw), {"block_size": 16, "use_sobel": True, "use_consistency_check": False}),
    ("census+consistency",    lambda l, r, **kw: compute_disparity_enhanced(l, r, method="CENSUS", **kw), {"block_size": 16, "use_sobel": False, "use_consistency_check": True}),
    ("census+sobel+consistency",    lambda l, r, **kw: compute_disparity_enhanced(l, r, method="CENSUS", **kw), {"block_size": 16, "use_sobel": True, "use_consistency_check": True}),

]

def colorize_disparity_jet(disp8u: np.ndarray) -> np.ndarray:
    """Colorize an 8-bit disparity map using the JET colormap (most common)."""
    disp8u = disp8u.astype(np.uint8, copy=False)
    return cv2.applyColorMap(disp8u, cv2.COLORMAP_JET)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def process_stereo_pair(left_path, right_path, block_size=5, max_disparity=64, out_dir="outputs_enhanced"):
    """
    Process a stereo image pair with all disparity methods (including the improved one),
    visualize, and save results.

    Saves: <out_dir>/<pair_name>/<pair_name>_<method>.png           (8-bit)
           <out_dir>/<pair_name>/<pair_name>_<method>_color.png    (JET colorized)
           <out_dir>/<pair_name>/<pair_name>_overview.png          (panel of all color maps)
    """
    print(f"\nProcessing stereo pair: {os.path.basename(left_path)} & {os.path.basename(right_path)}")

    left_image = cv2.imread(left_path)
    right_image = cv2.imread(right_path)

    if left_image is None or right_image is None:
        print(f"Error: Could not load images from {left_path} or {right_path}")
        return

    pair_name = os.path.splitext(os.path.basename(left_path))[0]
    save_root = os.path.join(out_dir, pair_name)
    ensure_dir(save_root)

    # Run all methods and collect results
    results = []  # list of (method_name, disp8u, elapsed_s)
    for name, func, overrides in METHODS:
        # allow per-method overrides but keep global defaults otherwise
        bs = overrides.get("block_size", block_size)
        md = overrides.get("max_disparity", max_disparity)
        
        # Build kwargs: merge overrides with block_size and max_disparity
        kwargs = {**overrides, "block_size": bs, "max_disparity": md}

        t0 = time.time()
        disp8u = func(left_image, right_image, **kwargs)
        elapsed = time.time() - t0

        if disp8u is None or disp8u.ndim != 2:
            print(f"[WARN] {name}: expected single-channel 8-bit disparity; got {None if disp8u is None else disp8u.shape}")
            continue
        if disp8u.dtype != np.uint8:
            disp8u = disp8u.astype(np.uint8)

        results.append((name, disp8u, elapsed))
        print(f"  {name.upper():>15}: {elapsed:.3f}s, block={bs}, D={md}, "
              f"range [min={int(disp8u.min())}, max={int(disp8u.max())}]")

        # Save grayscale map
        gray_path = os.path.join(save_root, f"{pair_name}_{name}.png")
        cv2.imwrite(gray_path, disp8u)

        # Save JET colorized map
        color_map = colorize_disparity_jet(disp8u)  # BGR
        color_path = os.path.join(save_root, f"{pair_name}_{name}_color.png")
        cv2.imwrite(color_path, color_map)

    # Overview panel (JET color) saved as one PNG
    if results:
        cols = min(4, len(results))  # Increased from 3 to 4 columns for better layout
        rows = int(np.ceil(len(results) / cols))
        plt.figure(figsize=(4 * cols, 3.5 * rows))
        for i, (name, disp8u, _) in enumerate(results, 1):
            plt.subplot(rows, cols, i)
            jet_bgr = colorize_disparity_jet(disp8u)
            jet_rgb = cv2.cvtColor(jet_bgr, cv2.COLOR_BGR2RGB)
            plt.imshow(jet_rgb)
            plt.title(name.upper())
            plt.axis('off')
        plt.tight_layout()
        overview_path = os.path.join(save_root, f"{pair_name}_overview.png")
        plt.savefig(overview_path, dpi=150)
        plt.close()
        print(f"  Saved overview panel → {overview_path}")

if __name__ == "__main__":
    process_stereo_pair('img/corridorl.jpg',   'img/corridorr.jpg',   block_size=5, max_disparity=64)
    process_stereo_pair('img/triclopsi2l.jpg', 'img/triclopsi2r.jpg', block_size=5, max_disparity=64)
