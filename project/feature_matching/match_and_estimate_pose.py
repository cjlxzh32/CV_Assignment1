import os
import glob
import json
import argparse
import numpy as np
import cv2 as cv
from pathlib import Path
import open3d as o3d
import csv


def load_features(scene_dir):
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


def match_features(desc1, desc2, method="FLANN", ratio_thresh=0.7, max_matches=500):
    """
    Args:
        desc1: First descriptor array
        desc2: Second descriptor array
        method: "FLANN" or "BF"
        ratio_thresh: Lowe ratio threshold
        max_matches: optional limit for final matches count
    Returns:
        good_matches: filtered matches
    """
    if method == "FLANN":
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desc1.astype(np.float32), desc2.astype(np.float32), k=2)
    else:
        bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
        matches = bf.match(desc1, desc2)

        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:max_matches]
        return good_matches

    good_matches = [m for m, n in matches if m.distance < ratio_thresh * n.distance]

    if len(good_matches) > max_matches:
        good_matches = sorted(good_matches, key=lambda x: x.distance)[:max_matches]

    return good_matches


def compute_reproj_error(P1, P2, pts1, pts2, pts3d):
    pts3d_h = np.hstack([pts3d, np.ones((pts3d.shape[0], 1))])
    proj1 = (P1 @ pts3d_h.T).T
    proj2 = (P2 @ pts3d_h.T).T
    proj1 = proj1[:, :2] / proj1[:, 2:3]
    proj2 = proj2[:, :2] / proj2[:, 2:3]
    err1 = np.linalg.norm(proj1 - pts1, axis=1)
    err2 = np.linalg.norm(proj2 - pts2, axis=1)
    rmse = np.sqrt(np.mean(np.hstack([err1, err2]) ** 2))
    return float(rmse)


def estimate_pose(kps1, kps2, matches, K, scene=""):
    """
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

    if scene == "chinese_heritage_centre":
        F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC, 1.5, 0.99)
    elif scene == "arch":
        F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC, 1.2, 0.95)
    else:
        F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC, 1.0, 0.99)

    if F is None or mask is None:
        raise ValueError("findFundamentalMat failed (no F returned)")

    inliers1 = pts1[mask.ravel() == 1]
    inliers2 = pts2[mask.ravel() == 1]
    inlier_count = len(inliers1)
    inlier_ratio = inlier_count / max(len(pts1), 1)

    if inlier_count < 5:
        raise ValueError(f"Too few inliers ({inlier_count}) after RANSAC")

    if scene == "chinese_heritage_centre":
        E, mask = cv.findEssentialMat(pts1, pts2, K, cv.RANSAC, 0.999, 1.5)
    elif scene == "arch":
        E, mask = cv.findEssentialMat(pts1, pts2, K, cv.RANSAC, 0.95, 1.2)
    else:
        E, mask = cv.findEssentialMat(pts1, pts2, K, cv.RANSAC, 0.999, 1.0)

    try:
        num_inliers, R, t, mask_pose = cv.recoverPose(E, inliers1, inliers2, K)
    except cv.error as e:
        raise ValueError(f"recoverPose failed: {e}")

    if num_inliers < 5:
        raise ValueError("recoverPose returned too few inliers (0)")

    return F, E, R, t, inliers1, inliers2, inlier_count, inlier_ratio


def validate_camera_pose(R, t, prev_poses):
    if len(prev_poses) > 0:
        last_R, last_t = prev_poses[-1]
        angle_diff = np.arccos(min(1.0, max(-1.0, (np.trace(np.dot(R.T, last_R)) - 1) / 2))) * 180 / np.pi
        trans_diff = np.linalg.norm(t - last_t)

        if angle_diff > 30 or trans_diff > 5:
            print(f"[WARN] Camera pose change too large: angle={angle_diff:.2f}°, translation={trans_diff:.2f}")
            return False

    return True


def visualize_matches(img1, img2, kps1, kps2, matches, out_path):
    kp_cv1 = [cv.KeyPoint(x=f[0], y=f[1], size=f[2]) for f in kps1]
    kp_cv2 = [cv.KeyPoint(x=f[0], y=f[1], size=f[2]) for f in kps2]
    img_matches = cv.drawMatches(
        img1, kp_cv1, img2, kp_cv2, matches, None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv.imwrite(out_path, img_matches)


def save_results(scene_out, img1, img2, matches, F, E, R, t, inlier_count, inlier_ratio, reproj_rmse,
                 is_non_sequential=False):
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

    if is_non_sequential:
        out_json = scene_out / "non_sequential" / "matches" / f"{img1}_{img2}_pose.json"
    else:
        out_json = scene_out / "matches" / f"{img1}_{img2}_pose.json"

    with open(out_json, "w") as f:
        json.dump(out_data, f, indent=2)
    return out_json


def save_pose(R, t, save_path):
    """Save rotation and translation to .txt"""
    np.savetxt(save_path, np.hstack([R.flatten(), t.flatten()])[None], fmt="%.6f")


def save_inlier_points(scene_out, img1, img2, inliers1, inliers2, is_non_sequential=False):
    """Save inlier points to .npz"""
    if is_non_sequential:
        out_path = scene_out / "non_sequential" / "matches" / f"{img1}_{img2}_inliers.npz"
    else:
        out_path = scene_out / "matches" / f"{img1}_{img2}_inliers.npz"

    np.savez_compressed(out_path, pts1=inliers1, pts2=inliers2)
    return out_path


def accumulate_camera_poses(pose_list, scene_name=""):
    cam_centers = []
    T_world = np.eye(4)
    cam_centers.append(T_world[:3, 3].copy())

    # For the chinese_heritage_centre and arch, regularly reset the accumulation
    reset_interval = 0
    if scene_name == "chinese_heritage_centre" or scene_name == "arch":
        reset_interval = 5

    reset_points = []
    if reset_interval > 0:
        reset_points = list(range(reset_interval, len(pose_list), reset_interval))

    for i, (R, t) in enumerate(pose_list):
        # Create relative transformation matrix
        T_rel = np.eye(4)
        T_rel[:3, :3] = R
        T_rel[:3, 3] = t.flatten()

        # Invert to get camera_i+1 pose relative to camera_i
        T_rel_inv = np.eye(4)
        T_rel_inv[:3, :3] = R.T
        T_rel_inv[:3, 3] = -R.T @ t.flatten()

        if reset_interval > 0 and i in reset_points:
            last_R = T_world[:3, :3]
            T_world = np.eye(4)
            T_world[:3, :3] = last_R

        T_world = T_world @ T_rel_inv
        cam_centers.append(T_world[:3, 3].copy())
    return np.array(cam_centers)


def visualize_camera_trajectory(cam_centers, scene_out, scene_name, show=False):
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
    vis.capture_screen_image(str(scene_out / "viz" / f"trajectory_{scene_name}.png"))
    if show:
        vis.run()
    vis.destroy_window()
    print(f"[INFO] Saved trajectory image to {scene_out}/viz/trajectory_{scene_name}.png")


def check_projection_consistency(pts3d, cameras, img_width, img_height):
    projection_counts = np.zeros(pts3d.shape[0])
    for P in cameras:
        pts3d_h = np.hstack([pts3d, np.ones((pts3d.shape[0], 1))])
        proj = (P @ pts3d_h.T).T
        proj = proj[:, :2] / proj[:, 2:3]
        valid = ((proj[:, 0] >= 0) & (proj[:, 0] < img_width) &
                 (proj[:, 1] >= 0) & (proj[:, 1] < img_height))
        projection_counts += valid

    return np.mean(projection_counts > 1)


def adjust_K_for_orientation(w, h, K):
    """Adjust camera intrinsic parameters based on image orientation"""
    if w > h:  # Horizontal image
        K_use = np.array([
            [K[1, 1], 0, K[1, 2]],
            [0, K[0, 0], K[0, 2]],
            [0, 0, 1]
        ])
    else:
        K_use = K.copy()
    return K_use


def create_directory_structure(scene_out):
    (scene_out / "matches").mkdir(parents=True, exist_ok=True)
    (scene_out / "poses").mkdir(parents=True, exist_ok=True)
    (scene_out / "viz").mkdir(parents=True, exist_ok=True)

    non_seq_dir = scene_out / "non_sequential"
    non_seq_dir.mkdir(exist_ok=True)
    (non_seq_dir / "matches").mkdir(exist_ok=True)
    (non_seq_dir / "poses").mkdir(exist_ok=True)

    return {
        "sequential": {
            "matches": scene_out / "matches",
            "poses": scene_out / "poses",
            "viz": scene_out / "viz"
        },
        "non_sequential": {
            "matches": non_seq_dir / "matches",
            "poses": non_seq_dir / "poses",
        }
    }


def try_match_image_pair(img1, img2, keypoints, descriptors, K, scene, img_dir, scene_out, dirs,
                         ratio_thresh, max_matches, min_inliers, cameras, is_non_sequential=False):
    kps1, desc1 = keypoints[img1], descriptors[img1]
    kps2, desc2 = keypoints[img2], descriptors[img2]

    if desc1.size == 0 or desc2.size == 0:
        print(f"[WARN] Empty descriptors for {img1} or {img2}, skipping")
        return None

    img1c = cv.imread(str(img_dir / f"{img1}.jpg"))
    img2c = cv.imread(str(img_dir / f"{img2}.jpg"))
    if img1c is None or img2c is None:
        print(f"[ERR] Failed to load {img1} or {img2}, skipping")
        return None

    h1, w1 = img1c.shape[:2]
    h2, w2 = img2c.shape[:2]
    K1 = adjust_K_for_orientation(w1, h1, K)
    K2 = adjust_K_for_orientation(w2, h2, K)
    K_use = K1
    print(f"[INFO] Trying match {img1} ↔ {img2}")
    matches = match_features(desc1, desc2, method="FLANN",
                             ratio_thresh=ratio_thresh,
                             max_matches=max_matches)

    if len(matches) < 30:
        print(f"[WARN] Too few raw matches ({len(matches)}), skipping")
        return None

    try:
        result = estimate_pose(kps1, kps2, matches, K_use, scene)
        if result is None:
            return None
        F, E, R, t, inliers1, inliers2, inlier_count, inlier_ratio = result
    except Exception as e:
        print(f"[ERR] Pose estimation failed for {img1}-{img2}: {e}")
        return None

    if inlier_count < min_inliers:
        print(f"[WARN] Too few inliers ({inlier_count}), skipping")
        return None

    if len(cameras) > 0 and not validate_camera_pose(R, t, cameras):
        print(f"[WARN] Camera pose validation failed for {img1}-{img2}, skipping")
        return None

    save_inlier_points(scene_out, img1, img2, inliers1, inliers2, is_non_sequential=is_non_sequential)

    P1 = K_use @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K_use @ np.hstack((R, t))
    pts4d = cv.triangulatePoints(P1, P2, inliers1.T, inliers2.T)
    pts3d = (pts4d[:3] / pts4d[3]).T
    reproj_rmse = compute_reproj_error(P1, P2, inliers1, inliers2, pts3d)

    if not is_non_sequential:
        vis_path = scene_out / "viz" / f"{img1}_{img2}_matches.jpg"
        visualize_matches(img1c, img2c, kps1, kps2, matches, str(vis_path))

    if len(cameras) > 0:
        try:
            prev_R, prev_t = cameras[-1]
            prev_P = K_use @ np.hstack((prev_R, prev_t))
            consistency = check_projection_consistency(pts3d, [prev_P, P2], w1, h1)
            print(f"[INFO] Projection consistency for {img1}-{img2}: {consistency:.2f}")
        except Exception as e:
            print(f"[WARN] Could not check projection consistency: {e}")

    return {
        "img1": img1,
        "img2": img2,
        "matches": matches,
        "F": F,
        "E": E,
        "R": R,
        "t": t,
        "inliers1": inliers1,
        "inliers2": inliers2,
        "inlier_count": inlier_count,
        "inlier_ratio": inlier_ratio,
        "reproj_rmse": reproj_rmse,
        "pts3d": pts3d,
        "P1": P1,
        "P2": P2
    }


def process_scene(scene, K, project_dir, output_dir, min_inliers=50, process_from_start=True, show=False):
    summary_records = []
    scene_path = project_dir / "feature_extraction" / "features" / scene
    keypoints, descriptors = load_features(scene_path)
    images = sorted(list(keypoints.keys()))

    print(f"[INFO] Found {len(images)} images in keypoints dictionary")
    print(f"[INFO] First 5 image names: {images[:5] if len(images) >= 5 else images}")
    print(f"[INFO] Last 5 image names: {images[-5:] if len(images) >= 5 else images}")

    scene_out = output_dir / scene
    dirs = create_directory_structure(scene_out)

    pose_list = []
    cameras = []

    img_dir = project_dir / "scenes" / "images" / scene

    # Adaptively adjust parameters based on scene names
    max_matches = 500
    ratio_thresh = 0.7
    error_threshold = 7.0
    if scene == "chinese_heritage_centre":
        max_matches = 2000
        ratio_thresh = 0.65
        min_inliers = 40
    elif scene == "arch":
        max_matches = 1500
        ratio_thresh = 0.8
        min_inliers = 40
    print(
        f"[INFO] Scene '{scene}' settings: max_matches={max_matches}, ratio_thresh={ratio_thresh}, min_inliers={min_inliers}, error_threshold={error_threshold}")


    i = 0 if process_from_start else 0
    n = len(images)

    non_sequential_matches = []
    for idx in range(0, n - 5, 5):
        non_sequential_matches.append((idx, idx + 5))

    while i < n - 1:
        current_idx = i
        max_attempt_idx = min(i + 10, n)
        match_found = False
        for j in range(i + 1, max_attempt_idx):
            img1, img2 = images[i], images[j]
            result = try_match_image_pair(img1, img2, keypoints, descriptors, K, scene,
                                          img_dir, scene_out, dirs, ratio_thresh, max_matches,
                                          min_inliers, cameras, is_non_sequential=False)
            if result is None:
                print(f"[INFO] Failed to match {img1}-{img2}, trying next frame...")
                continue
            if result["reproj_rmse"] > error_threshold:
                print(
                    f"[WARN] High reprojection error ({result['reproj_rmse']:.2f}) for {img1}-{img2}, trying next frame...")
                continue
            match_found = True

            save_results(scene_out, img1, img2, result["matches"], result["F"], result["E"],
                         result["R"], result["t"], result["inlier_count"], result["inlier_ratio"],
                         result["reproj_rmse"], is_non_sequential=False)

            save_pose(result["R"], result["t"], dirs["sequential"]["poses"] / f"{img1}_{img2}.txt")

            pose_list.append((result["R"], result["t"]))
            cameras.append((result["R"], result["t"]))
            summary_records.append([
                scene,
                f"{img1}-{img2}",
                len(result["matches"]),
                result["inlier_count"],
                round(result["inlier_ratio"], 3),
                round(result["reproj_rmse"], 3),
                "OK"
            ])

            print(
                f"[DONE] Match {img1}-{img2}: inliers={result['inlier_count']}, ratio={result['inlier_ratio']:.2f}, RMSE={result['reproj_rmse']:.2f}")

            i = j
            break

        if not match_found:
            print(
                f"[WARN] Failed to find a good match for frame {images[i]} after {max_attempt_idx - i - 1} attempts, skipping to next frame.")
            i += 1

    # Processing non continuous frame matching
    non_seq_summary = []
    if non_sequential_matches:
        print(f"[INFO] Processing non-sequential matches for {scene}...")
        for idx1, idx5 in non_sequential_matches:
            if idx1 >= len(images) or idx5 >= len(images):
                continue

            img1, img5 = images[idx1], images[idx5]
            result = try_match_image_pair(
                img1, img5, keypoints, descriptors, K, scene, img_dir, scene_out, dirs,
                ratio_thresh, max_matches, min_inliers, [], is_non_sequential=True
            )

            if result is not None:
                save_results(scene_out, img1, img5, result["matches"], result["F"], result["E"],
                             result["R"], result["t"], result["inlier_count"], result["inlier_ratio"],
                             result["reproj_rmse"], is_non_sequential=True)

                save_pose(result["R"], result["t"], dirs["non_sequential"]["poses"] / f"{img1}_{img5}.txt")

                non_seq_summary.append([
                    scene,
                    f"{img1}-{img5} (non-sequential)",
                    len(result["matches"]),
                    result["inlier_count"],
                    round(result["inlier_ratio"], 3),
                    round(result["reproj_rmse"], 3),
                    "OK"
                ])

                print(
                    f"[DONE] Non-sequential match {img1}-{img5}: inliers={result['inlier_count']}, ratio={result['inlier_ratio']:.2f}, RMSE={result['reproj_rmse']:.2f}")

    csv_path = os.path.join(output_dir, f"{scene}_pose_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scene", "pair", "num_matches", "inlier_count", "inlier_ratio", "reproj_rmse", "status"])
        writer.writerows(summary_records)

    if non_seq_summary:
        non_seq_csv = os.path.join(output_dir, f"{scene}_non_sequential_summary.csv")
        with open(non_seq_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["scene", "pair", "num_matches", "inlier_count", "inlier_ratio", "reproj_rmse", "status"])
            writer.writerows(non_seq_summary)

    print(f"[INFO] Scene '{scene}' summary saved to {csv_path}")
    success_count = sum(1 for r in summary_records if r[-1] == "OK")
    print(f"[INFO] Successful pairs: {success_count}/{len(summary_records) if len(summary_records) > 0 else 0} "
          f"({100 * success_count / len(summary_records) if len(summary_records) > 0 else 0:.1f}%)")

    avg_keypoints = np.mean([len(kps) for kps in keypoints.values()])

    if summary_records:
        avg_inlier_ratio = np.mean([record[4] for record in summary_records])
        avg_reproj_error = np.mean([record[5] for record in summary_records if record[5] != "-"])
        print(
            f"[STATS] {scene} - Avg Keypoints: {avg_keypoints:.1f}, Avg Inlier Ratio: {avg_inlier_ratio:.3f}, Avg Reproj Error: {avg_reproj_error:.3f}")
    else:
        print(f"[STATS] {scene} - Avg Keypoints: {avg_keypoints:.1f}, No matches found")

    if len(pose_list) > 0:
        cam_centers = accumulate_camera_poses(pose_list, scene_name=scene)
        visualize_camera_trajectory(cam_centers, scene_out, scene_name=scene, show=show)
        if scene == "chinese_heritage_centre" or scene == "arch":
            cam_centers_alt = accumulate_camera_poses(pose_list)
            visualize_camera_trajectory(cam_centers_alt, scene_out, scene_name=f"{scene}_alt", show=show)


def main():
    parser = argparse.ArgumentParser(description="Feature Matching & Pose Estimation")
    parser.add_argument("--scene", type=str, default=None, help="Scene name (default: all scenes)")
    parser.add_argument("--min_inliers", type=int, default=50, help="Minimum number of inliers to accept a pair")
    parser.add_argument("--param_set", type=str, default="default",
                        choices=["default", "strict", "relaxed"],
                        help="Parameter set to use for matching and pose estimation")
    parser.add_argument("--force_start", action="store_true", help="Force start from the first frame")
    parser.add_argument("--output", type=str, default=None, help="Custom output directory name")
    parser.add_argument("--error_threshold", type=float, default=None,
                        help="Maximum acceptable reprojection error (default: 7.0, arch: 9.0)")
    parser.add_argument("--show", action="store_true", help="Show Open3D visualization")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parents[1]
    calib_path = project_dir / "calibration" / "results" / "iphone_calibration.npz"
    scenes_dir = project_dir / "feature_extraction" / "features"

    if args.output:
        output_dir = project_dir / "feature_matching" / args.output
    elif args.param_set != "default":
        output_dir = project_dir / "feature_matching" / f"results_{args.param_set}"
    else:
        output_dir = project_dir / "feature_matching" / "results"

    os.makedirs(output_dir, exist_ok=True)

    data = np.load(calib_path, allow_pickle=True)
    K = data["K"]
    print("[INFO] Loaded K:\n", K)

    all_scenes = sorted(os.listdir(scenes_dir))
    target_scenes = [args.scene] if args.scene else all_scenes

    for scene in target_scenes:
        print(f"\n=== Processing scene: {scene} ===")
        process_scene(scene, K, project_dir, output_dir,
                      min_inliers=args.min_inliers,
                      process_from_start=args.force_start,
                      show=args.show)
    print("\n=== ALL SCENES COMPLETE ===")


if __name__ == "__main__":
    main()