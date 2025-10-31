from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
import open3d as o3d
import csv
import cv2 as cv
import json

from config_paths import (
    C_OUTPUT_DIR,
    VOXEL_SIZE,
    REPROJ_THRESH_PX,
    MIN_POINTS_PER_PAIR_FOR_MERGE,
)
from load_inputs import load_pair_inliers


def make_projection_matrix(K, R, t):
    Rt = np.hstack([R, t])  # (3,4)
    return K @ Rt

def project_points(K, R, t, X3d):
    Rt = np.hstack([R, t])
    Xh = np.hstack([X3d, np.ones((X3d.shape[0],1))]).T
    uvw = K @ (Rt @ Xh)
    uv = (uvw[:2] / uvw[2:]).T
    return uv

def is_in_front(R, t, X3d):
    Xc = (R @ X3d.T + t).T
    return Xc[:, 2] > 1e-6

def undistort_points_px(pts_px, K, dist):
    if dist is None or np.allclose(dist, 0):
        return pts_px.astype(np.float64)
    pts = pts_px.reshape(-1, 1, 2).astype(np.float64)
    pts_ud = cv.undistortPoints(pts, K, dist, P=K)
    return pts_ud.reshape(-1, 2)

def triangulate_pair(K, RA, tA, RB, tB, ptsA_px, ptsB_px):
    P_A = make_projection_matrix(K, RA, tA)
    P_B = make_projection_matrix(K, RB, tB)
    X4d = cv.triangulatePoints(P_A, P_B, ptsA_px.T, ptsB_px.T)
    X3d = (X4d[:3] / X4d[3]).T
    return X3d

def triangulation_angles_deg(K, RA, RB, ptsA_px, ptsB_px):
    Kinv = np.linalg.inv(K)
    ones = np.ones((ptsA_px.shape[0], 1), dtype=np.float64)
    a_cam = (Kinv @ np.hstack([ptsA_px, ones]).T).T
    b_cam = (Kinv @ np.hstack([ptsB_px, ones]).T).T
    a_cam /= np.linalg.norm(a_cam, axis=1, keepdims=True)
    b_cam /= np.linalg.norm(b_cam, axis=1, keepdims=True)
    a_world = (RA.T @ a_cam.T).T
    b_world = (RB.T @ b_cam.T).T
    a_world /= np.linalg.norm(a_world, axis=1, keepdims=True)
    b_world /= np.linalg.norm(b_world, axis=1, keepdims=True)
    dots = np.sum(a_world * b_world, axis=1).clip(-1.0, 1.0)
    ang = np.degrees(np.arccos(dots))
    return ang


def save_poses_initial(poses_global: Dict[str, Tuple[np.ndarray, np.ndarray]], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    poses_dict = {}
    for name, (R_w2c, t_w2c) in poses_global.items():
        poses_dict[name] = {
            "R_w2c": np.asarray(R_w2c, dtype=float).tolist(),
            "t_w2c": np.asarray(t_w2c, dtype=float).reshape(3).tolist()
        }
    with open(out_dir / "poses_initial.json", "w") as f:
        json.dump(poses_dict, f, indent=2)
    print(f"[OK] wrote {out_dir/'poses_initial.json'}")

def save_tracks_initial(tracks_points: dict, out_dir: Path):

    out_dir.mkdir(parents=True, exist_ok=True)
    points = []
    observations = []
    for pid, data in tracks_points.items():
        points.append({"id": int(pid), "X": np.asarray(data["X"], float).tolist()})
        for (img, uv) in data["obs"]:
            observations.append({"cam": img, "pt_id": int(pid), "uv": [float(uv[0]), float(uv[1])]})
    js = {"points": points, "observations": observations}
    with open(out_dir / "tracks_initial.json", "w") as f:
        json.dump(js, f, indent=2)
    print(f"[OK] wrote {out_dir/'tracks_initial.json'}  (Stage-1 3D + tracks)")


def reconstruct_sparse_scene(
    K: np.ndarray,
    dist: np.ndarray,
    poses_global: Dict[str, Tuple[np.ndarray, np.ndarray]],
    pairs: List[tuple],
    summary_rows: List[dict],
    angle_thresh_deg: float = 3.0,
    merge_radius_px: float = 2.0,
):

    all_points_accum = []
    metrics_rows = []


    tracks_points = {}
    img_obs_index = {}
    next_pid = 0

    iter_rows = summary_rows if summary_rows else [{"pair": f"{a}-{b}"} for (a,b) in pairs]

    for row in iter_rows:
        pair_name = row["pair"]
        if "-" not in pair_name:
            continue
        imgA, imgB = pair_name.split("-")
        imgA, imgB = imgA.strip(), imgB.strip()

        if imgA not in poses_global or imgB not in poses_global:
            print(f"[SKIP] {pair_name}: missing pose in global graph")
            continue

        ptsA_px, ptsB_px = load_pair_inliers(imgA, imgB)
        if ptsA_px is None or ptsA_px.size == 0:
            print(f"[SKIP] {pair_name}: no inlier file or empty matches")
            continue


        ptsA_id_px = undistort_points_px(ptsA_px, K, dist)
        ptsB_id_px = undistort_points_px(ptsB_px, K, dist)

        RA, tA = poses_global[imgA]
        RB, tB = poses_global[imgB]


        X_all = triangulate_pair(K, RA, tA, RB, tB, ptsA_id_px, ptsB_id_px)


        uvA_hat = project_points(K, RA, tA, X_all)
        uvB_hat = project_points(K, RB, tB, X_all)
        errA = np.linalg.norm(uvA_hat - ptsA_id_px, axis=1)
        errB = np.linalg.norm(uvB_hat - ptsB_id_px, axis=1)
        reproj_ok = (errA < REPROJ_THRESH_PX) & (errB < REPROJ_THRESH_PX)
        front_ok  = is_in_front(RA, tA, X_all) & is_in_front(RB, tB, X_all)
        ang_deg = triangulation_angles_deg(K, RA, RB, ptsA_id_px, ptsB_id_px)
        ang_ok = (ang_deg > angle_thresh_deg)
        keep_mask = reproj_ok & front_ok & ang_ok

        X_keep = X_all[keep_mask]
        kept = X_keep.shape[0]
        raw  = X_all.shape[0]

        med_ang = float(np.median(ang_deg)) if ang_deg.size else float("nan")
        mean_err = float("nan")
        if kept > 0:
            mean_err = 0.5 * (np.mean(errA[keep_mask]) + np.mean(errB[keep_mask]))
        print(f"[PAIR] {pair_name}: raw={raw}, kept={kept}, mean_reproj={mean_err if kept>0 else 'nan'}")
        if ang_deg.size:
            p10 = float(np.percentile(ang_deg, 10))
            p90 = float(np.percentile(ang_deg, 90))
            print(f"       ang(deg): median={med_ang:.2f}, p10={p10:.2f}, p90={p90:.2f}")

        metrics_rows.append({
            "pair": pair_name,
            "raw_points": raw,
            "kept_points": kept,
            "mean_reproj_err_px": mean_err,
            "median_angle_deg": med_ang,
        })


        if kept > 0:
            kept_indices = np.where(keep_mask)[0]
            for idx in kept_indices:
                uvA = ptsA_id_px[idx]
                uvB = ptsB_id_px[idx]
                Xw  = X_all[idx]

                assigned_pid = None
                if imgA in img_obs_index:
                    for (pid_exist, uv_exist) in img_obs_index[imgA]:
                        if np.linalg.norm(uv_exist - uvA) < merge_radius_px:
                            assigned_pid = pid_exist
                            break
                if assigned_pid is None and imgB in img_obs_index:
                    for (pid_exist, uv_exist) in img_obs_index[imgB]:
                        if np.linalg.norm(uv_exist - uvB) < merge_radius_px:
                            assigned_pid = pid_exist
                            break

                if assigned_pid is None:
                    pid = next_pid
                    next_pid += 1
                    tracks_points[pid] = {"X": Xw.copy(), "obs": [(imgA, uvA.copy()), (imgB, uvB.copy())]}
                    img_obs_index.setdefault(imgA, []).append((pid, uvA.copy()))
                    img_obs_index.setdefault(imgB, []).append((pid, uvB.copy()))
                else:
                    pid = assigned_pid
                    def add_obs(img, uv):
                        if not any((o_img==img and np.linalg.norm(o_uv-uv)<1e-3)
                                   for (o_img,o_uv) in tracks_points[pid]["obs"]):
                            tracks_points[pid]["obs"].append((img, uv.copy()))
                            img_obs_index.setdefault(img, []).append((pid, uv.copy()))
                    add_obs(imgA, uvA)
                    add_obs(imgB, uvB)
                    tracks_points[pid]["X"] = 0.5 * (tracks_points[pid]["X"] + Xw)


            if kept >= MIN_POINTS_PER_PAIR_FOR_MERGE:
                all_points_accum.append(X_keep)
            else:
                thr = 0.8 * REPROJ_THRESH_PX
                reproj_ok2 = (errA < thr) & (errB < thr)
                ang_ok2 = (ang_deg > (angle_thresh_deg + 0.5))
                keep2 = reproj_ok2 & front_ok & ang_ok2
                X_keep2 = X_all[keep2]
                if X_keep2.shape[0] > 0:
                    print(f"[WEAK->KEEP {pair_name}] kept={kept}<{MIN_POINTS_PER_PAIR_FOR_MERGE}, strict_keep={X_keep2.shape[0]}; merged.")
                    all_points_accum.append(X_keep2)
                else:
                    print(f"[DROP WEAK {pair_name}] kept=0 on stricter pass.")
        else:
            print(f"[DROP EMPTY {pair_name}] kept=0")


    if len(all_points_accum) > 0:
        all_points_accum = np.vstack(all_points_accum)
    else:
        all_points_accum = np.zeros((0, 3), dtype=np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points_accum)
    pcd_ds = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)


    out_dir = Path(C_OUTPUT_DIR) / "initial"
    out_dir.mkdir(parents=True, exist_ok=True)

    ply_path = out_dir / "sparse_points_initial.ply"
    o3d.io.write_point_cloud(str(ply_path), pcd_ds)
    print(f"[OK] wrote {ply_path} with {len(pcd_ds.points)} points")

    metrics_csv = out_dir / "sparse_metrics.csv"
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["pair","raw_points","kept_points","mean_reproj_err_px","median_angle_deg"]
        )
        writer.writeheader()
        writer.writerows(metrics_rows)
    print(f"[OK] wrote {metrics_csv}")

    return tracks_points, np.asarray(pcd_ds.points), metrics_rows
