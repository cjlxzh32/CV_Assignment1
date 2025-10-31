import json
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np

from config_paths import C_OUTPUT_DIR, load_calibration


THIS = Path(__file__).resolve().parent
SCENE_DIR = Path(C_OUTPUT_DIR)            
INITIAL_DIR = SCENE_DIR / "initial"

POSES_JSON = INITIAL_DIR / "poses_initial.json"
TRACKS_JSON = INITIAL_DIR / "tracks_initial.json"

OUT_JSON = SCENE_DIR / "ba_problem_export.json"
OUT_NPZ  = SCENE_DIR / "ba_problem_export.npz"



def _as_R(x):
    arr = np.asarray(x, dtype=float)
    if arr.size == 9:
        arr = arr.reshape(3,3)
    assert arr.shape == (3,3), f"R wrong shape: {arr.shape}"
    return arr

def _quat_to_R_any(q):
    q = np.asarray(q, dtype=float).reshape(-1)
    assert q.size == 4
    def toR(a,b,c,d):
        n = np.sqrt(a*a+b*b+c*c+d*d) + 1e-12
        a,b,c,d = a/n, b/n, c/n, d/n
        return np.array([
            [1-2*(c*c+d*d), 2*(b*c - d*a), 2*(b*d + c*a)],
            [2*(b*c + d*a), 1-2*(b*b+d*d), 2*(c*d - b*a)],
            [2*(b*d - c*a), 2*(c*d + b*a), 1-2*(b*b+c*c)]
        ], dtype=float)
    R1 = toR(q[0], q[1], q[2], q[3])  
    R2 = toR(q[3], q[0], q[1], q[2])  
    e1 = np.linalg.norm(R1.T@R1 - np.eye(3))
    e2 = np.linalg.norm(R2.T@R2 - np.eye(3))
    return R1 if e1 <= e2 else R2

def load_stage1_poses(poses_json_path: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    data = json.load(open(poses_json_path, "r"))

    def lk(d): return {k.lower(): k for k in d.keys()}
    def pick(d, *cands):
        m = lk(d)
        for c in cands:
            if c.lower() in m: return d[m[c.lower()]]
        return None

    poses = {}
    if isinstance(data, dict):
        items = list(data.items())
    elif isinstance(data, list):
        items = []
        for i, v in enumerate(data):
            name = pick(v, "name", "id", "stem") or f"cam_{i:04d}"
            items.append((name, v))
    else:
        raise ValueError("poses_initial.json 顶层必须是 dict 或 list")

    for name, v in items:
        try:
            R = pick(v, "R_w2c", "R_wc", "R", "rotation", "rot", "R_world2cam")
            t = pick(v, "t_w2c", "t_wc", "t", "translation", "trans", "t_world2cam")
            if R is not None and t is not None:
                poses[name] = (_as_R(R), np.asarray(t, float).reshape(3,1)); continue
            Rc = pick(v, "R_c2w", "Rcw", "R_cam2world", "R_cam_to_world")
            tc = pick(v, "t_c2w", "tcw", "t_cam2world", "t_cam_to_world")
            if Rc is not None and tc is not None:
                Rc = _as_R(Rc); tc = np.asarray(tc, float).reshape(3,1)
                Rw = Rc.T; tw = -Rc.T @ tc
                poses[name] = (Rw, tw); continue
            q = pick(v, "q", "quat", "quaternion", "q_wxyz", "q_xyzw")
            if q is not None:
                Rw2c = _quat_to_R_any(q)
                t_wc = pick(v, "t_w2c", "t_wc", "t", "translation")
                if t_wc is not None:
                    poses[name] = (Rw2c, np.asarray(t_wc, float).reshape(3,1)); continue
                t_cw = pick(v, "t_c2w", "tcw")
                if t_cw is not None:
                    tc = np.asarray(t_cw, float).reshape(3,1)
                    Rw = Rw2c; tw = -Rw.T @ tc
                    poses[name] = (Rw, tw); continue
            raise KeyError("No recognizable pose keys")
        except Exception as e:
            raise KeyError(f"[{name}] cannot parse pose. keys={list(v.keys())}. err={e}")
    return poses



def main():

    if not POSES_JSON.exists():
        raise FileNotFoundError(f"Missing {POSES_JSON}")
    if not TRACKS_JSON.exists():
        raise FileNotFoundError(f"Missing {TRACKS_JSON} (run Stage-1 to produce tracks_initial.json)")


    K, _ = load_calibration()
    K = K.astype(float)


    poses_global = load_stage1_poses(POSES_JSON)   
    cam_names_sorted = sorted(poses_global.keys())
    cam_id_of = {name: idx for idx, name in enumerate(cam_names_sorted)}


    tracks = json.load(open(TRACKS_JSON, "r"))
    points_list = tracks["points"]
    observations_raw = tracks["observations"]


    cameras = []
    for name in cam_names_sorted:
        R_w2c, t_w2c = poses_global[name]
        R_c2w = R_w2c.T
        t_c2w = -R_w2c.T @ t_w2c
        cameras.append({
            "name": name,
            "id": cam_id_of[name],
            "R_w2c": R_w2c.tolist(),
            "t_w2c": t_w2c.reshape(3).tolist(),
            "R_c2w": R_c2w.tolist(),
            "t_c2w": t_c2w.reshape(3).tolist()
        })

    points = [{"id": int(p["id"]), "X": [float(p["X"][0]), float(p["X"][1]), float(p["X"][2])]}
              for p in points_list]

    observations: List[dict] = []
    used_cams = set()
    for o in observations_raw:
        img_name = o["cam"]
        if img_name not in cam_id_of:
            continue
        observations.append({
            "cam_id": cam_id_of[img_name],
            "pt_id": int(o["pt_id"]),
            "uv": [float(o["uv"][0]), float(o["uv"][1])]
        })
        used_cams.add(cam_id_of[img_name])


    used_cam_ids = sorted(list(used_cams))
    if len(used_cam_ids) == 0:
        raise RuntimeError("No cameras referenced by observations.")
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
            "convention": "x_cam = R_w2c * x_world + t_w2c",
            "fixed_cam_id": 0
        }
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(ba_problem, f, indent=2)
    print(f"[OK] wrote {OUT_JSON}")


    cam_ids = np.array([c["id"] for c in cameras], dtype=np.int32)
    cam_names_arr = np.array([c["name"] for c in cameras])

    R_w2c = np.stack([np.array(c["R_w2c"], float) for c in cameras])  # (Nc,3,3)
    t_w2c = np.stack([np.array(c["t_w2c"], float) for c in cameras])  # (Nc,3)
    R_c2w = np.stack([np.array(c["R_c2w"], float) for c in cameras])
    t_c2w = np.stack([np.array(c["t_c2w"], float) for c in cameras])

    pts_ids = np.array([p["id"] for p in points], dtype=np.int32)
    pts_xyz = np.stack([np.array(p["X"], float) for p in points])     # (Np,3)

    obs_cam = np.array([o["cam_id"] for o in observations], dtype=np.int32)
    obs_pid = np.array([o["pt_id"] for o in observations], dtype=np.int32)
    obs_uv  = np.stack([np.array(o["uv"], float) for o in observations])  # (No,2)

    np.savez_compressed(
        OUT_NPZ,
        K=K,
        cam_ids=cam_ids,
        cam_names=cam_names_arr,
        R_w2c=R_w2c,
        t_w2c=t_w2c,
        R_c2w=R_c2w,
        t_c2w=t_c2w,
        pts_ids=pts_ids,
        pts_xyz=pts_xyz,
        obs_cam=obs_cam,
        obs_pid=obs_pid,
        obs_uv=obs_uv
    )
    print(f"[OK] wrote {OUT_NPZ}")
    print("[DONE] Export complete. You can now run ceres_ba_runner.")

if __name__ == "__main__":
    main()
