import os
import glob
import numpy as np
import cv2 as cv
import csv

# --------- Config ---------
FRAMES_DIR = "test_video"                     # input images
OUT_DIR = "features"          # where to save features
CALIB_PATH = "calibration/results/iphone_calibration.npz"  # optional; set to None to skip undistort
GLOB_PATTERNS = ["*.jpg","*.jpeg","*.png","*.JPG","*.PNG"]
VISUALIZE_FIRST_N = 5                 # draw keypoints for the first N images (0 = disable)
USE_ORB_IF_SIFT_MISSING = True
# --------------------------

os.makedirs(OUT_DIR, exist_ok=True)
vis_dir = os.path.join(OUT_DIR, "viz")
os.makedirs(vis_dir, exist_ok=True)

# Load calibration (optional)
K, dist, img_size = None, None, None
if CALIB_PATH and os.path.exists(CALIB_PATH):
    data = np.load(CALIB_PATH, allow_pickle=True)
    K = data["K"]
    dist = data["dist"]
    img_size = tuple(data["img_size"])
    print(f"[INFO] Loaded calibration from {CALIB_PATH}")
else:
    print("[INFO] No calibration file found or path disabled; skipping undistort.")

# Collect image paths
paths = []
for pat in GLOB_PATTERNS:
    paths.extend(glob.glob(os.path.join(FRAMES_DIR, pat)))
paths = sorted(paths)
if not paths:
    raise SystemExit(f"[ERR] No images found in {FRAMES_DIR}")

print(f"[INFO] Found {len(paths)} images.")

# Create detector/descriptor
detector_name = "SIFT"
try:
    sift = cv.SIFT_create()
    extractor = sift
except Exception as e:
    if USE_ORB_IF_SIFT_MISSING:
        print("[WARN] SIFT not available. Falling back to ORB.")
        extractor = cv.ORB_create(nfeatures=4000)
        detector_name = "ORB"
    else:
        raise SystemExit("[ERR] SIFT not available and fallback disabled.")

# Helper: undistort
def maybe_undistort(img):
    if K is None or dist is None:
        return img
    h, w = img.shape[:2]
    newK, _ = cv.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0)
    return cv.undistort(img, K, dist, None, newK)

# Summary CSV
csv_path = os.path.join(OUT_DIR, "feature_summary.csv")
csv_file = open(csv_path, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["filename", "detector", "num_keypoints", "descriptor_shape"])

# Process images
for idx, p in enumerate(paths, 1):
    img = cv.imread(p, cv.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[WARN] Cannot read: {p}")
        continue

    img_ud = maybe_undistort(img)

    # detect & compute
    kps, desc = extractor.detectAndCompute(img_ud, None)

    # Save features
    base = os.path.splitext(os.path.basename(p))[0]
    np.save(os.path.join(OUT_DIR, f"{base}_keypoints.npy"),
            np.array([kp.pt + (kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
                      for kp in kps], dtype=np.float32))
    # Note: kp array columns = [x, y, size, angle, response, octave, class_id]

    if desc is None:
        # Some images may yield no descriptors (e.g., too blurry/blank)
        desc = np.empty((0, 128 if detector_name=="SIFT" else extractor.getDescriptorSize()), dtype=np.float32)
    np.save(os.path.join(OUT_DIR, f"{base}_descriptors.npy"), desc)

    # Log
    csv_writer.writerow([
        os.path.basename(p),
        detector_name,
        len(kps),
        list(desc.shape)
    ])

    # Optional visualization for first N images
    if VISUALIZE_FIRST_N and idx <= VISUALIZE_FIRST_N:
        img_color = cv.cvtColor(img_ud, cv.COLOR_GRAY2BGR)
        vis = cv.drawKeypoints(img_color, kps, None,
                               flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv.imwrite(os.path.join(vis_dir, f"{base}_kps.jpg"), vis)

    if idx % 10 == 0 or idx == len(paths):
        print(f"[INFO] Processed {idx}/{len(paths)}")

csv_file.close()

# Write a little README for teammates
with open(os.path.join(OUT_DIR, "README_features.txt"), "w") as f:
    f.write(
        "Features extracted for all images in ./frames\n"
        f"- Detector: {detector_name}\n"
        f"- Calibration used: {'Yes' if K is not None else 'No'}\n"
        f"- Summary CSV: feature_summary.csv\n"
        f"- Keypoints: *_keypoints.npy (columns: x, y, size, angle, response, octave, class_id)\n"
        f"- Descriptors: *_descriptors.npy (shape: [N, D])\n"
        "Consumers (Group B): load matching image pairs, match descriptors (FLANN/BF),\n"
        "then use K (from results/iphone_calibration.npz) for recoverPose/triangulate.\n"
    )

print(f"[DONE] Saved features to {OUT_DIR}")
