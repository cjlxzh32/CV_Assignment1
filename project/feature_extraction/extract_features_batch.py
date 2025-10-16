#!/usr/bin/env python3
"""
Batch feature extraction script for all 3 scene folders
"""

import os
import glob
import numpy as np
import cv2 as cv
import csv

def extract_features_for_folder(frames_dir, out_dir, calib_path=None, visualize_first_n=5):
    """
    Extract features for all images in a folder
    
    Args:
        frames_dir: Path to folder containing images
        out_dir: Path to save features
        calib_path: Path to calibration file (optional)
        visualize_first_n: Number of images to visualize (0 = disable)
    """
    
    print(f"\n=== Processing {frames_dir} ===")
    
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    vis_dir = os.path.join(out_dir, "viz")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Load calibration (optional)
    K, dist, img_size = None, None, None
    if calib_path and os.path.exists(calib_path):
        data = np.load(calib_path, allow_pickle=True)
        K = data["K"]
        dist = data["dist"]
        img_size = tuple(data["img_size"])
        print(f"[INFO] Loaded calibration from {calib_path}")
    else:
        print("[INFO] No calibration file found or path disabled; skipping undistort.")
    
    # Collect image paths
    glob_patterns = ["*.jpg","*.jpeg","*.png","*.JPG","*.PNG"]
    paths = []
    for pat in glob_patterns:
        paths.extend(glob.glob(os.path.join(frames_dir, pat)))
    paths = sorted(paths)
    
    if not paths:
        print(f"[ERR] No images found in {frames_dir}")
        return
    
    print(f"[INFO] Found {len(paths)} images.")
    
    # Create detector/descriptor
    detector_name = "SIFT"
    try:
        sift = cv.SIFT_create()
        extractor = sift
    except Exception as e:
        print("[WARN] SIFT not available. Falling back to ORB.")
        extractor = cv.ORB_create(nfeatures=4000)
        detector_name = "ORB"
    
    # Helper: undistort
    def maybe_undistort(img):
        if K is None or dist is None:
            return img
        h, w = img.shape[:2]
        newK, _ = cv.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0)
        return cv.undistort(img, K, dist, None, newK)
    
    # Summary CSV
    csv_path = os.path.join(out_dir, "feature_summary.csv")
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
        np.save(os.path.join(out_dir, f"{base}_keypoints.npy"),
                np.array([kp.pt + (kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
                          for kp in kps], dtype=np.float32))
        
        if desc is None:
            # Some images may yield no descriptors (e.g., too blurry/blank)
            desc = np.empty((0, 128 if detector_name=="SIFT" else extractor.getDescriptorSize()), dtype=np.float32)
        np.save(os.path.join(out_dir, f"{base}_descriptors.npy"), desc)
        
        # Log
        csv_writer.writerow([
            os.path.basename(p),
            detector_name,
            len(kps),
            list(desc.shape)
        ])
        
        # Optional visualization for first N images
        if visualize_first_n and idx <= visualize_first_n:
            img_color = cv.cvtColor(img_ud, cv.COLOR_GRAY2BGR)
            vis = cv.drawKeypoints(img_color, kps, None,
                                   flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv.imwrite(os.path.join(vis_dir, f"{base}_kps.jpg"), vis)
        
        if idx % 10 == 0 or idx == len(paths):
            print(f"[INFO] Processed {idx}/{len(paths)}")
    
    csv_file.close()
    
    # Write README
    with open(os.path.join(out_dir, "README_features.txt"), "w") as f:
        f.write(
            f"Features extracted for all images in {frames_dir}\n"
            f"- Detector: {detector_name}\n"
            f"- Calibration used: {'Yes' if K is not None else 'No'}\n"
            f"- Summary CSV: feature_summary.csv\n"
            f"- Keypoints: *_keypoints.npy (columns: x, y, size, angle, response, octave, class_id)\n"
            f"- Descriptors: *_descriptors.npy (shape: [N, D])\n"
            "Consumers: load matching image pairs, match descriptors (FLANN/BF),\n"
            "then use K (from calibration) for recoverPose/triangulate.\n"
        )
    
    print(f"[DONE] Saved features to {out_dir}")

def main():
    """Main function to process all image folders"""
    
    # Get script directory and set relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    # Base paths (relative to project directory)
    base_scenes_path = os.path.join(project_dir, "scenes", "images")
    base_features_path = os.path.join(script_dir, "features")
    calib_path = os.path.join(project_dir, "calibration", "results", "iphone_calibration.npz")
    
    # Define folders to process
    folders = ['arch', 'chinese_heritage_centre', 'pavilion']
    
    for folder in folders:
        frames_dir = os.path.join(base_scenes_path, folder)
        out_dir = os.path.join(base_features_path, folder)
        
        if os.path.exists(frames_dir):
            extract_features_for_folder(frames_dir, out_dir, calib_path, visualize_first_n=5)
        else:
            print(f"[ERR] Folder not found: {frames_dir}")
    
    print(f"\n=== ALL PROCESSING COMPLETE ===")
    print(f"Features saved to: {base_features_path}")

if __name__ == "__main__":
    main()
