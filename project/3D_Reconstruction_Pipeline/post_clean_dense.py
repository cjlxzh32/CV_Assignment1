import open3d as o3d
import numpy as np
from pathlib import Path

in_ply  = "result/arch/dense_model/dense.ply"
out_ply = "result/arch/dense_model/dense_clean.ply"

pcd = o3d.io.read_point_cloud(in_ply)
n0 = len(pcd.points)
if n0 == 0:
    raise RuntimeError(f"Input has 0 points: {in_ply}")


nn = np.asarray(pcd.compute_nearest_neighbor_distance())
mean_nn = float(np.mean(nn)) if nn.size > 0 else 0.0

bbox = pcd.get_axis_aligned_bounding_box()
diag = float(np.linalg.norm(bbox.get_extent()))

nb_neighbors = 20 if n0 < 200000 else 30
std_ratio    = 2.0 if n0 < 200000 else 1.5

radius   = max(3.0 * mean_nn, 1e-6) if mean_nn > 0 else max(0.01 * diag, 1e-6)
nb_points = 3 if n0 < 200000 else 5

print(f"[INFO] points={n0}, mean_nn={mean_nn:.6f}, diag={diag:.6f}")
print(f"[INFO] StatisticalOutlier: nb_neighbors={nb_neighbors}, std_ratio={std_ratio}")
print(f"[INFO] RadiusOutlier: nb_points={nb_points}, radius={radius:.6f}")


pcd_s, ind_s = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
pcd_s = pcd.select_by_index(ind_s)
n1 = len(pcd_s.points)
print(f"[STEP1] statistical -> {n1}/{n0}")


def try_radius_filter(pcd_in, nb_pts, rad):
    p, ind = pcd_in.remove_radius_outlier(nb_points=nb_pts, radius=rad)
    return pcd_in.select_by_index(ind)

pcd_r = try_radius_filter(pcd_s, nb_points, radius)
n2 = len(pcd_r.points)
if n2 == 0:

    pcd_r = try_radius_filter(pcd_s, max(2, nb_points-1), radius*2.0)
    n2 = len(pcd_r.points)
    print(f"[STEP2] radius (relaxed) -> {n2}/{n1}")
else:
    print(f"[STEP2] radius -> {n2}/{n1}")


if n2 == 0:
    print("[WARN] radius filter removed all points, using statistical-only result.")
    pcd_r = pcd_s
    n2 = n1


voxel = max(2.0 * mean_nn, 1e-6) if mean_nn > 0 else 0.005 * diag
pcd_v = pcd_r.voxel_down_sample(voxel_size=voxel)
n3 = len(pcd_v.points)
print(f"[STEP3] voxel({voxel:.6f}) -> {n3}/{n2}")


final_pcd = pcd_v if n3 > 0 else (pcd_r if n2 > 0 else pcd_s)
nf = len(final_pcd.points)
if nf == 0:
    raise RuntimeError("All filters produced 0 points, aborting save.")

Path(out_ply).parent.mkdir(parents=True, exist_ok=True)
o3d.io.write_point_cloud(out_ply, final_pcd)
print(f"[DONE] Saved: {out_ply} | before: {n0} after: {nf}")
