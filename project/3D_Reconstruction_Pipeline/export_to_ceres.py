import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from config_paths import C_OUTPUT_DIR, load_calibration

PROJECT_ROOT = Path(__file__).resolve().parent
ARCH_DIR = PROJECT_ROOT / "result" / "arch"
INITIAL_DIR = ARCH_DIR / "initial"

POSES_JSON = INITIAL_DIR / "poses_initial.json"
TRACKS_JSON = INITIAL_DIR / "tracks_initial.json"
OUT_DIR = ARCH_DIR


def load_stage1_poses(poses_json_path: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    data = json.load(open(poses_json_path, "r"))
    poses = {}
    if isinstance(data, dict):
        for name, v in data.items():
            R = np.asarray(v["R_w2c"], dtype=float).reshape(3,3)
            t = np.asarray(v["t_w2c"], dtype=float).reshape(3,1)
            poses[name] = (R, t)
    else:
        raise ValueError("poses_initial.json 顶层应为 dict")
    return poses

def export_ba_from_stage1_tracks(K, poses_global, tracks_json_path: Path, out_dir: Path):
    tj = json.load(open(tracks_json_path, "r"))


    cam_names = sorted(poses_global.keys())
    cam_id_of = {name: i for i, name in enumerate(cam_names)}


    cameras = []
    for name in cam_names:
        R_w2c, t_w2c = poses_global[name]
        R_c2w = R_w2c.T
        t_c2w = -R_w2c.T @ t_w2c.reshape(3, 1)
        cameras.append({
            "name": name,
            "id": cam_id_of[name],
            "R_w2c": R_w2c.tolist(),
            "t_w2c": t_w2c.reshape(3).tolist(),
            "R_c2w": R_c2w.tolist(),
            "t_c2w": t_c2w.reshape(3).tolist()
        })


    points = []
    for p in tj["points"]:
        points.append({"id": int(p["id"]), "X": list(map(float, p["X"]))})


    observations = []
    for o in tj["observations"]:
        observations.append({
            "cam_id": cam_id_of[o["cam"]],
            "pt_id": int(o["pt_id"]),
            "uv": [float(o["uv"][0]), float(o["uv"][1])]
        })


    used_cam_ids = sorted(set(obs["cam_id"] for obs in observations))

    if len(used_cam_ids) == 0:
        raise RuntimeError("No cameras referenced by observations. Cannot export BA.")

    cam_old2new = {old: new for new, old in enumerate(used_cam_ids)}

    cameras_used = []
    for c in cameras:
        if c["id"] in cam_old2new:
            cc = dict(c)
            cc["id"] = cam_old2new[c["id"]]
            cameras_used.append(cc)

    for obs in observations:
        obs["cam_id"] = cam_old2new[obs["cam_id"]]

    cameras = cameras_used


    ba_problem = {
        "K": K.tolist(),
        "cameras": cameras,
        "points": points,
        "observations": observations,
        "meta": {
            "num_cameras": len(cameras),
            "num_points": len(points),
            "num_observations": len(observations),
            "fixed_cam_id": 0,  # 第 0 台现在肯定有观测
            "note": "cameras reordered to ensure camera 0 has observations"
        }
    }


    ba_json = {
        "K": np.asarray(K, float).tolist(),
        "cameras": cameras,
        "points": points,
        "observations": observations,
        "meta": {
            "num_cameras": len(cameras),
            "num_points": len(points),
            "num_observations": len(observations),
            "convention": "R_w2c means x_cam = R_w2c * x_world + t_w2c",
            "source": "Stage-1 triangulation reused (no re-triangulation)"
        }
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    jpath = out_dir / "ba_problem_export.json"
    with open(jpath, "w") as f:
        json.dump(ba_json, f, indent=2)
    print(f"[OK] wrote {jpath}")


    cam_ids = np.array([c["id"] for c in cameras], dtype=np.int32)
    cam_names_arr = np.array([c["name"] for c in cameras])

    R_w2c_arr = np.stack([np.array(c["R_w2c"], float) for c in cameras])  # (Nc,3,3)
    t_w2c_arr = np.stack([np.array(c["t_w2c"], float) for c in cameras])  # (Nc,3)
    R_c2w_arr = np.stack([np.array(c["R_c2w"], float) for c in cameras])
    t_c2w_arr = np.stack([np.array(c["t_c2w"], float) for c in cameras])

    pts_ids = np.array([p["id"] for p in points], dtype=np.int32)
    pts_xyz = np.stack([np.array(p["X"], float) for p in points])

    obs_cam = np.array([o["cam_id"] for o in observations], dtype=np.int32)
    obs_pid = np.array([o["pt_id"] for o in observations], dtype=np.int32)
    obs_uv  = np.stack([np.array(o["uv"], float) for o in observations])

    npz_path = out_dir / "ba_problem_export.npz"
    np.savez_compressed(
        npz_path,
        K=np.asarray(K, float),
        cam_ids=cam_ids,
        cam_names=cam_names_arr,
        R_w2c=R_w2c_arr,
        t_w2c=t_w2c_arr,
        R_c2w=R_c2w_arr,
        t_c2w=t_c2w_arr,
        pts_ids=pts_ids,
        pts_xyz=pts_xyz,
        obs_cam=obs_cam,
        obs_pid=obs_pid,
        obs_uv=obs_uv
    )
    print(f"[OK] wrote {npz_path}")
    print("[DONE] Export complete. Feed into Ceres BA.")

def main():

    K, _ = load_calibration()
    K = np.asarray(K, dtype=np.float64)


    if not POSES_JSON.exists():
        raise FileNotFoundError(f"Missing {POSES_JSON}")
    poses_global = load_stage1_poses(POSES_JSON)
    print(f"[INFO] loaded poses: {len(poses_global)} cameras")


    if not TRACKS_JSON.exists():
        raise FileNotFoundError(f"Missing {TRACKS_JSON} (run sparse_reconstruction.py first)")
    export_ba_from_stage1_tracks(K, poses_global, TRACKS_JSON, OUT_DIR)

if __name__ == "__main__":
    main()
