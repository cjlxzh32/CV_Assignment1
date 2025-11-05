from load_inputs import load_summary_pairs
from poses_global import accumulate_global_poses
from sparse_reconstruction import (
    reconstruct_sparse_scene,
    save_poses_initial,
    save_tracks_initial,
    save_observations_only, 
)
from config_paths import load_calibration, C_OUTPUT_DIR
from pathlib import Path
import numpy as np


def main():

    K, dist = load_calibration()
    K = K.astype(np.float64)
    dist = dist.astype(np.float64).ravel()
    if np.allclose(dist, 0):
        print("[INFO] Using portrait intrinsics (no rotation of K). Distortion is zeros.")
    print("K =\n", K)
    print("dist =", dist)


    pairs, rows = load_summary_pairs()
    poses_global = accumulate_global_poses(pairs)
    print(f"[INFO] poses_global has {len(poses_global)} cameras")

    names_sorted = sorted(poses_global.keys())
    for i, img in enumerate(names_sorted[:6]):
        R, t = poses_global[img]
        t = t.reshape(3)
        print(f"  {i:02d} {img}: t={t} |t|={np.linalg.norm(t):.3f}")


    tracks_points, _, _ = reconstruct_sparse_scene(
        K=K,
        dist=dist,
        poses_global=poses_global,
        pairs=pairs,
        summary_rows=rows,
        angle_thresh_deg=2.0,  
    )


    initial_dir = Path(C_OUTPUT_DIR) / "initial"
    save_poses_initial(poses_global, initial_dir)
    save_tracks_initial(tracks_points, initial_dir)
    save_observations_only(tracks_points, initial_dir) 

    print(f"[DONE] Stage 1 complete. Outputs written to {initial_dir}")

if __name__ == "__main__":
    main()
