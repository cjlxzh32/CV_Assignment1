"""
Task B - Feature Matching & Pose Estimation
"""

import os
import glob
import json
import argparse
import numpy as np
import cv2 as cv
from pathlib import Path
import open3d as o3d
from itertools import combinations
import csv


def load_features(scene_dir):
    """Load keypoints and descriptors from extract_features_batch.py outputs"""
    keypoints, descriptors = {}, {}
    for kp_path in sorted(glob.glob(os.path.join(scene_dir, "*_keypoints.npy"))):
        base = Path(kp_path).stem.replace("_keypoints", "")
        desc_path = os.path.join(scene_dir, f"{base}_descriptors.npy")
        if not os.path.exists(desc_path):
            continue
        keypoints[base] = np.load(kp_path)
        descriptors[base] = np.load(desc_path)

    if len(keypoints) < 2:
        raise ValueError(f"Not enough images found in {scene_dir}. Need at least 2 images.")

    return keypoints, descriptors


def match_features(desc1, desc2, method="FLANN", ratio_thresh=0.6, max_matches=500):
    """
    Match feature descriptors
    Args:
        desc1: First descriptor array
        desc2: Second descriptor array
        method: "FLANN" or "BF"
        ratio_thresh: Lowe ratio threshold
        max_matches: optional limit for final matches count
    Returns:
        good_matches: filtered matches (list of cv.DMatch)
        mask: RANSAC inlier mask (if applied)
    """
    if method == "FLANN":
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desc1.astype(np.float32), desc2.astype(np.float32), k=2)
    else:
        bf = cv.BFMatcher()
        matches = bf.knnMatch(desc1, desc2, k=2)

    good_matches = [m for m, n in matches if m.distance < ratio_thresh * n.distance]

    if len(good_matches) > max_matches:
    # sort by descriptor distance
        good_matches = sorted(good_matches, key=lambda x: x.distance)[:max_matches]

    return good_matches


def compute_reproj_error(P1, P2, pts1, pts2, pts3d):
    """Compute mean reprojection RMSE in both images"""
    pts3d_h = np.hstack([pts3d, np.ones((pts3d.shape[0], 1))])
    proj1 = (P1 @ pts3d_h.T).T
    proj2 = (P2 @ pts3d_h.T).T
    proj1 = proj1[:, :2] / proj1[:, 2:3]
    proj2 = proj2[:, :2] / proj2[:, 2:3]
    err1 = np.linalg.norm(proj1 - pts1, axis=1)
    err2 = np.linalg.norm(proj2 - pts2, axis=1)
    rmse = np.sqrt(np.mean(np.hstack([err1, err2]) ** 2))
    return float(rmse)


def estimate_pose(kps1, kps2, matches, K):
    """
    Compute F, E, R, t using matched keypoints
    Returns:
        F: Fundamental matrix
        E: Essential matrix
        R: Rotation matrix (from camera1 to camera2)
        t: Translation vector (from camera1 to camera2)
        inlier_pts1: Inlier points in image1
        inlier_pts2: Inlier points in image2
    """
    pts1 = np.float32([kps1[m.queryIdx][:2] for m in matches])
    pts2 = np.float32([kps2[m.trainIdx][:2] for m in matches])

    F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC, 1.0, 0.99)
    if F is None or mask is None:
        raise ValueError("findFundamentalMat failed (no F returned)")

    inliers1 = pts1[mask.ravel() == 1]
    inliers2 = pts2[mask.ravel() == 1]
    inlier_count = len(inliers1)
    inlier_ratio = inlier_count / max(len(pts1), 1)

    if inlier_count < 5:
        raise ValueError(f"Too few inliers ({inlier_count}) after RANSAC")

    #E = K.T @ F @ K
    E, mask = cv.findEssentialMat(pts1, pts2, K, cv.RANSAC, 0.999, 1.0)

    try:
        num_inliers, R, t, mask_pose = cv.recoverPose(E, inliers1, inliers2, K)
    except cv.error as e:
        raise ValueError(f"recoverPose failed: {e}")

    if num_inliers < 5:
        raise ValueError("recoverPose returned too few inliers (0)")

    return F, E, R, t, inliers1, inliers2, inlier_count, inlier_ratio


def visualize_matches(img1, img2, kps1, kps2, matches, out_path):
    """Visualize matched keypoints"""
    kp_cv1 = [cv.KeyPoint(x=f[0], y=f[1], size=f[2]) for f in kps1]
    kp_cv2 = [cv.KeyPoint(x=f[0], y=f[1], size=f[2]) for f in kps2]
    img_matches = cv.drawMatches(
        img1, kp_cv1, img2, kp_cv2, matches, None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv.imwrite(out_path, img_matches)


def save_results(scene_out, img1, img2, matches, F, E, R, t, inlier_count, inlier_ratio, reproj_rmse):
    out_data = {
        "image_1": img1,
        "image_2": img2,
        "num_matches": len(matches),
        "inlier_count": inlier_count,
        "inlier_ratio": round(inlier_ratio, 4),
        "reproj_rmse": round(reproj_rmse, 4),
        "F": F.tolist(),
        "E": E.tolist(),
        "R": R.tolist(),
        "t": t.tolist()
    }
    out_json = scene_out / "matches" / f"{img1}_{img2}_pose.json"
    with open(out_json, "w") as f:
        json.dump(out_data, f, indent=2)
    return out_json


def save_pose(R, t, save_path):
    """Save rotation and translation to .txt"""
    np.savetxt(save_path, np.hstack([R.flatten(), t.flatten()])[None], fmt="%.6f")


def accumulate_camera_poses(pose_list):
    """
    Accumulate camera poses to compute camera trajectory in world coordinates
    Args:
        pose_list: List of (R, t) tuples where R and t are from camera_i to camera_i+1
    Returns:
        cam_centers: Nx3 array of camera centers in world coordinates
    """
    cam_centers = []
    # First camera is at world origin
    T_world = np.eye(4)
    cam_centers.append(T_world[:3, 3].copy())

    for R, t in pose_list:
        # R, t represent transformation from camera_i to camera_i+1
        # We need to invert this to get camera_i+1's pose in world coordinates

        # Create relative transformation matrix (camera_i to camera_i+1)
        T_rel = np.eye(4)
        T_rel[:3, :3] = R
        T_rel[:3, 3] = t.flatten()

        # Invert to get camera_i+1 pose relative to camera_i
        T_rel_inv = np.eye(4)
        T_rel_inv[:3, :3] = R.T
        T_rel_inv[:3, 3] = -R.T @ t.flatten()

        # Accumulate: T_world = T_world @ T_rel_inv
        T_world = T_world @ T_rel_inv

        # Extract camera center
        cam_centers.append(T_world[:3, 3].copy())

    return np.array(cam_centers)


def visualize_camera_trajectory(cam_centers, scene_out, scene_name, show=False):
    """Visualize camera trajectory using Open3D"""
    pts = o3d.utility.Vector3dVector(cam_centers)
    lines = [[i, i + 1] for i in range(len(cam_centers) - 1)]
    colors = [[0, 0, 1] for _ in lines]

    line_set = o3d.geometry.LineSet(points=pts, lines=o3d.utility.Vector2iVector(lines))
    line_set.colors = o3d.utility.Vector3dVector(colors)
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=show)
    vis.add_geometry(line_set)
    vis.add_geometry(coord)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(str(scene_out / f"trajectory_{scene_name}.png"))
    if show:
        vis.run()
    vis.destroy_window()
    print(f"[INFO] Saved trajectory image to {scene_out}/trajectory_{scene_name}.png")


def process_scene(scene, K, project_dir, output_dir, show=False, pair_mode="sequential", min_inliers=50):
    summary_records = []
    scene_path = project_dir / "feature_extraction" / "features" / scene
    keypoints, descriptors = load_features(scene_path)
    images = sorted(list(keypoints.keys()))

    scene_out = output_dir / scene
    (scene_out / "matches").mkdir(parents=True, exist_ok=True)
    (scene_out / "poses").mkdir(parents=True, exist_ok=True)
    (scene_out / "viz").mkdir(parents=True, exist_ok=True)

    pose_list = []

    if pair_mode == "full":
        pairs = list(combinations(range(len(images)), 2))
    else:
        pairs = [(i, i + 1) for i in range(len(images) - 1)]

    for i, j in pairs:
        img1, img2 = images[i], images[j]
        kps1, desc1 = keypoints[img1], descriptors[img1]
        kps2, desc2 = keypoints[img2], descriptors[img2]
        if desc1.size == 0 or desc2.size == 0:
            continue

        matches = match_features(desc1, desc2)
        if len(matches) < 30:
            continue

        try:
            result = estimate_pose(kps1, kps2, matches, K)
            if result is None:
                continue
            F, E, R, t, inliers1, inliers2, inlier_count, inlier_ratio = result
        except Exception as e:
            print(f"[ERR] Pose estimation failed for {img1}-{img2}: {e}")
            continue

        if inlier_count < min_inliers:
            print(f"[WARN] Too few inliers ({inlier_count}), skipping {img1}-{img2}")
            summary_records.append([
                scene,
                f"{img1}-{img2}",
                0, 0, 0.0, "-", "FAIL"
            ])
            continue

        # Save inlier points
        np.savez_compressed(scene_out / "matches" / f"{img1}_{img2}_inliers.npz",
                            pts1=inliers1, pts2=inliers2)

        # Compute reprojection RMSE
        P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = K @ np.hstack((R, t))
        pts4d = cv.triangulatePoints(P1, P2, inliers1.T, inliers2.T)
        pts3d = (pts4d[:3] / pts4d[3]).T
        reproj_rmse = compute_reproj_error(P1, P2, inliers1, inliers2, pts3d)

        save_results(scene_out, img1, img2, matches, F, E, R, t, inlier_count, inlier_ratio, reproj_rmse)
        summary_records.append([
            scene,
            f"{img1}-{img2}",
            len(matches),
            inlier_count,
            round(inlier_ratio, 3),
            round(reproj_rmse, 3) if reproj_rmse is not None else "-",
            "OK"
        ])
        img_dir = project_dir / "scenes" / "images" / scene
        img1c, img2c = cv.imread(str(img_dir / f"{img1}.jpg")), cv.imread(str(img_dir / f"{img2}.jpg"))
        if img1c is not None and img2c is not None:
            vis_path = scene_out / "viz" / f"{img1}_{img2}_matches.jpg"
            visualize_matches(img1c, img2c, kps1, kps2, matches, str(vis_path))

        save_pose(R, t, scene_out / "poses" / f"{img1}_{img2}.txt")
        pose_list.append((R, t))

        print(f"[DONE] {img1}-{img2}: inliers={inlier_count}, ratio={inlier_ratio:.2f}, RMSE={reproj_rmse:.2f}")

    csv_path = os.path.join(output_dir, f"{scene}_pose_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scene", "pair", "num_matches", "inlier_count", "inlier_ratio", "reproj_rmse", "status"])
        writer.writerows(summary_records)

    print(f"\n[INFO] Scene '{scene}' summary saved to {csv_path}")
    success_count = sum(1 for r in summary_records if r[-1] == "OK")
    print(
        f"[INFO] Successful pairs: {success_count}/{len(summary_records)} ({100 * success_count / len(summary_records):.1f}%)")

    if len(pose_list) > 0:
        cam_centers = accumulate_camera_poses(pose_list)
        visualize_camera_trajectory(cam_centers, scene_out, scene_name=scene, show=show)


def main():
    parser = argparse.ArgumentParser(description="Feature Matching & Pose Estimation")
    parser.add_argument("--scene", type=str, default=None, help="Scene name (default: all scenes)")
    parser.add_argument("--show", action="store_true", help="Show Open3D visualization")
    parser.add_argument("--pair_mode", choices=["sequential", "full"], default="sequential",
                        help="Pairing strategy")
    parser.add_argument("--min_inliers", type=int, default=50, help="Minimum number of inliers to accept a pair")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parents[1]
    calib_path = project_dir / "calibration" / "results" / "iphone_calibration.npz"
    scenes_dir = project_dir / "feature_extraction" / "features"
    output_dir = project_dir / "feature_matching" / "results"
    os.makedirs(output_dir, exist_ok=True)

    data = np.load(calib_path, allow_pickle=True)
    K = data["K"]
    print("[INFO] Loaded K:\n", K)

    all_scenes = sorted(os.listdir(scenes_dir))
    target_scenes = [args.scene] if args.scene else all_scenes

    for scene in target_scenes:
        print(f"\n=== Processing scene: {scene} ===")
        process_scene(scene, K, project_dir, output_dir, show=args.show,
                      pair_mode=args.pair_mode, min_inliers=args.min_inliers)

    print("\n=== ALL SCENES COMPLETE ===")


if __name__ == "__main__":
    main()
