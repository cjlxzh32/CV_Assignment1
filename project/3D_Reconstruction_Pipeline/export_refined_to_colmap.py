import json, glob
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import cv2 as cv


try:
    from config_paths import load_calibration
except ImportError:
    def load_calibration():
        raise ImportError("Please provide load_calibration() in config_paths.py")


THIS_DIR = Path(__file__).resolve().parent
ARCH_RESULT_DIR = THIS_DIR / "result" / "arch"

REFINED_JSON = ARCH_RESULT_DIR / "ba_problem_ceres_refined.json"
ORIG_EXPORT_JSON = ARCH_RESULT_DIR / "ba_problem_export.json"
TRACKS_JSON = ARCH_RESULT_DIR / "initial" / "tracks_initial.json"

IMAGE_DIR = THIS_DIR / "../scenes/images/arch"

OUT_DIR = ARCH_RESULT_DIR / "colmap_model_refined"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def rotmat_to_colmap_quat(R_w2c: np.ndarray) -> Tuple[float, float, float, float]:

    R = np.asarray(R_w2c, dtype=float)
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S
    q = np.array([qw, qx, qy, qz], dtype=float)
    q /= (np.linalg.norm(q) + 1e-12)
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])

def detect_image_size(image_dir: Path) -> Tuple[int, int]:
    exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG", "*.JPEG"]
    imgs: List[str] = []
    for e in exts:
        imgs.extend(sorted(glob.glob(str(image_dir / e))))
    if not imgs:
        raise RuntimeError(f"[E] No images found in {image_dir}")
    im = cv.imread(imgs[0], cv.IMREAD_UNCHANGED)
    if im is None:
        raise RuntimeError(f"[E] Failed to read {imgs[0]}")
    h, w = im.shape[:2]
    return int(w), int(h)

def resolve_image_name(stem: str, image_dir: Path) -> str:
    exts = [".jpg", ".jpeg", ".png", ".JPG", ".PNG", ".JPEG"]
    for ex in exts:
        cand = image_dir / f"{stem}{ex}"
        if cand.exists():
            return cand.name
    return f"{stem}.jpg"


def write_cameras_txt_opencv(
    out_dir: Path,
    K: np.ndarray,
    dist: np.ndarray,
    img_w: int,
    img_h: int,
    cam_id_to_image_id: Dict[int, int],
):
    fx = float(K[0, 0]); fy = float(K[1, 1])
    cx = float(K[0, 2]); cy = float(K[1, 2])
    k1 = k2 = p1 = p2 = 0.0
    if dist is not None:
        d = np.asarray(dist, dtype=float).reshape(-1)
        if d.size >= 4:
            k1, k2, p1, p2 = d[0], d[1], d[2], d[3]

    lines = []
    lines.append("# Camera list with one line of data per camera:\n")
    lines.append("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
    lines.append("#   fx fy cx cy k1 k2 p1 p2 for OPENCV\n")

    for _, image_id in sorted(cam_id_to_image_id.items(), key=lambda x: x[1]):
        lines.append(
            f"{image_id} OPENCV {img_w} {img_h} "
            f"{fx} {fy} {cx} {cy} {k1} {k2} {p1} {p2}\n"
        )

    (out_dir / "cameras.txt").write_text("".join(lines), encoding="utf-8")
    print(f"[OK] wrote {out_dir/'cameras.txt'} (OPENCV)")

def write_images_txt_with_obs(
    out_dir: Path,
    refined: dict,
    cam_id_to_image_id: Dict[int, int],
    cam_id_to_stem: Dict[int, str],
    image_dir: Path,
    image_obs: Dict[int, List[Tuple[float, float, int]]],  # image_id -> list[(u,v,pid)]
):
    cam_dict = {int(c["id"]): c for c in refined["cameras_optimized"]}

    lines = []
    lines.append("# Image list with two lines of data per image:\n")
    lines.append("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
    lines.append("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")

    for cam_id, image_id in sorted(cam_id_to_image_id.items(), key=lambda x: x[1]):
        c = cam_dict[cam_id]
        R_w2c = np.array(c["R_w2c"], dtype=float)
        t_w2c = np.array(c["t_w2c"], dtype=float).reshape(3)
        qw, qx, qy, qz = rotmat_to_colmap_quat(R_w2c)
        stem = cam_id_to_stem[cam_id]
        img_name = resolve_image_name(stem, image_dir)


        lines.append(
            f"{image_id} {qw} {qx} {qy} {qz} {t_w2c[0]} {t_w2c[1]} {t_w2c[2]} {image_id} {img_name}\n"
        )


        obs_list = image_obs.get(image_id, [])
        if obs_list:
            chunks = [f"{u} {v} {pid}" for (u, v, pid) in obs_list]
            lines.append(" ".join(chunks) + "\n")
        else:
            lines.append("\n")

    (out_dir / "images.txt").write_text("".join(lines), encoding="utf-8")
    print(f"[OK] wrote {out_dir/'images.txt'} (with POINTS2D)")

def write_points3D_with_tracks(
    out_dir: Path,
    pt_xyz: Dict[int, np.ndarray],
    point_tracks: Dict[int, List[Tuple[int, int]]],  # pid -> list[(image_id, local_obs_idx)]
):
    lines = []
    lines.append("# 3D point list with one line of data per point:\n")
    lines.append("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")

    for pid in sorted(pt_xyz.keys()):
        X = pt_xyz[pid]
        base = [f"{pid} {X[0]} {X[1]} {X[2]} 255 255 255 1.0"]
        trk = point_tracks.get(pid, [])
        trk_chunks = []
        for (image_id, local_obs_idx) in trk:
            trk_chunks.append(f"{image_id} {local_obs_idx}")
        lines.append(" ".join(base + trk_chunks) + "\n")

    (out_dir / "points3D.txt").write_text("".join(lines), encoding="utf-8")
    print(f"[OK] wrote {out_dir/'points3D.txt'} (with TRACK)")


def main():

    if not REFINED_JSON.exists():
        raise FileNotFoundError(f"Missing {REFINED_JSON}")
    if not ORIG_EXPORT_JSON.exists():
        raise FileNotFoundError(f"Missing {ORIG_EXPORT_JSON}")
    if not TRACKS_JSON.exists():
        raise FileNotFoundError(f"Missing {TRACKS_JSON} (run Stage-1 to produce tracks_initial.json)")

    refined = json.load(open(REFINED_JSON, "r"))
    orig = json.load(open(ORIG_EXPORT_JSON, "r"))
    tracks = json.load(open(TRACKS_JSON, "r"))


    K, dist = load_calibration()
    K = np.asarray(K, dtype=float)
    if dist is not None:
        dist = np.asarray(dist, dtype=float)


    img_w, img_h = detect_image_size(IMAGE_DIR)
    print(f"[INFO] detected image size = {img_w} x {img_h}")


    cam_ids_sorted = sorted([int(c["id"]) for c in refined["cameras_optimized"]])
    cam_id_to_image_id = {cid: (i + 1) for i, cid in enumerate(cam_ids_sorted)}

    cam_id_to_stem: Dict[int, str] = {}
    for c in orig["cameras"]:
        cid = int(c["id"])
        stem = c.get("name", f"frame_{cid:03d}")
        cam_id_to_stem[cid] = stem


    name_to_image_id = {stem: cam_id_to_image_id[cid] for cid, stem in cam_id_to_stem.items() if cid in cam_id_to_image_id}


    valid_pt_ids = set(int(p["id"]) for p in tracks["points"])
    pt_xyz = {int(p["id"]): np.array(p["X"], dtype=float) for p in tracks["points"]}

    image_obs: Dict[int, List[Tuple[float, float, int]]] = {}   # image_id -> list[(u,v,pid)]
    for obs in tracks["observations"]:
        cam_name = obs["cam"]
        if cam_name not in name_to_image_id:
            continue
        image_id = int(name_to_image_id[cam_name])
        pid = int(obs["pt_id"])
        if pid not in valid_pt_ids:
            continue
        u, v = float(obs["uv"][0]), float(obs["uv"][1])
        image_obs.setdefault(image_id, []).append((u, v, pid))


    point_tracks: Dict[int, List[Tuple[int, int]]] = {}
    for image_id, obs_list in image_obs.items():
        for local_idx, (_, _, pid) in enumerate(obs_list):
            point_tracks.setdefault(pid, []).append((image_id, local_idx))


    write_cameras_txt_opencv(OUT_DIR, K, dist, img_w, img_h, cam_id_to_image_id)
    write_images_txt_with_obs(OUT_DIR, refined, cam_id_to_image_id, cam_id_to_stem, IMAGE_DIR, image_obs)
    write_points3D_with_tracks(OUT_DIR, pt_xyz, point_tracks)

    print(f"[DONE] Exported COLMAP model to {OUT_DIR}")

if __name__ == "__main__":
    main()
