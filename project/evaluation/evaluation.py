import json
import numpy as np

# -----------------------------
# 参数设置
# -----------------------------
ba_initial_path = "project/3D_Reconstruction_Pipeline/result/pavilion/ba_problem_export.json"
ba_refined_path = "project/3D_Reconstruction_Pipeline/result/pavilion/ba_problem_ceres_refined.json"

# -----------------------------
# 1. 加载数据
# -----------------------------
with open(ba_initial_path, 'r') as f:
    init_data = json.load(f)

with open(ba_refined_path, 'r') as f:
    refined_data = json.load(f)

K = np.array(refined_data["K"])
cameras_optimized = refined_data["cameras_optimized"]
points_optimized = refined_data["points_optimized"]
observations = init_data["observations"]

# 建立 pt_id -> 3D 点快速索引
points_dict = {pt['id']: np.array(pt['X']) for pt in points_optimized}

# 建立 cam_id -> 观测点列表快速索引
from collections import defaultdict
obs_by_cam = defaultdict(list)
for obs in observations:
    obs_by_cam[obs['cam_id']].append(obs)

# -----------------------------
# 2. 重投影误差函数
# -----------------------------
def reprojection_error_batch(X_worlds, uvs_obs, R_w2c, t_w2c, K):
    """
    批量计算重投影误差
    X_worlds: (N,3)
    uvs_obs: (N,2)
    R_w2c: (3,3)
    t_w2c: (3,)
    K: (3,3)
    返回: 每个点的重投影误差 (N,)
    """
    X_cam = (R_w2c @ X_worlds.T + t_w2c[:, None]).T  # (N,3)
    uv_proj_h = (K @ X_cam.T).T                       # (N,3)
    uv_proj = uv_proj_h[:, :2] / uv_proj_h[:, 2:3]
    errors = np.linalg.norm(uv_proj - uvs_obs, axis=1)
    return errors

# -----------------------------
# 3. 遍历相机计算误差
# -----------------------------
all_errors = []

for cam in cameras_optimized:
    cam_id = cam["id"]
    R = np.array(cam["R_w2c"])
    t = np.array(cam["t_w2c"])
    
    cam_obs = obs_by_cam[cam_id]
    if len(cam_obs) == 0:
        continue
    
    # 准备批量计算
    X_worlds = np.array([points_dict[obs['pt_id']] for obs in cam_obs])
    uvs_obs = np.array([obs['uv'] for obs in cam_obs])
    
    errors = reprojection_error_batch(X_worlds, uvs_obs, R, t, K)
    all_errors.extend(errors)
    
    print(f"Camera {cam_id}: {len(errors)} points, mean reprojection error = {errors.mean():.3f} px")

# -----------------------------
# 4. 全局平均误差
# -----------------------------
all_errors = np.array(all_errors)
print(f"\nOverall mean reprojection error: {all_errors.mean():.3f} px")
print(f"Overall median reprojection error: {np.median(all_errors):.3f} px")
print(f"Overall max reprojection error: {all_errors.max():.3f} px")


# import json
# import numpy as np
# import open3d as o3d
# import time
# import pandas as pd

# # -----------------------------
# # 参数设置
# # -----------------------------
# ba_initial_path = "project/3D_Reconstruction_Pipeline/result/pavilion/ba_problem_export.json"
# ba_refined_path = "project/3D_Reconstruction_Pipeline/result/pavilion/ba_problem_ceres_refined.json"
# dense_ply_path = "project/3D_Reconstruction_Pipeline/result/pavilion/dense_model/dense.ply"
# output_csv = "project/evaluation/metrics_summary.csv"

# # -----------------------------
# # 1. 加载稀疏点云与相机数据
# # -----------------------------
# with open(ba_initial_path, 'r') as f:
#     init_data = json.load(f)
    
# with open(ba_refined_path, 'r') as f:
#     refined_data = json.load(f)

# K = np.array(refined_data["K"]) # 相机内参，只有1个 refined_data["K"]=init_data["K"]
# cameras_optimized = refined_data["cameras_optimized"] # 优化后的不同相机位姿
# notes = refined_data["notes"] # 优化后的稀疏重建信息
# points_optimized = np.array(refined_data["points_optimized"]) # 稀疏点云
# observations = init_data["observations"] # 3d点投影回2d后对应的点和相机位姿信息

# # -----------------------------
# # 2. 计算重投影误差
# # -----------------------------
# def reprojection_error(X_world, uv_obs, R_w2c, t_w2c, K):
#     """
#     计算单个3D点的重投影误差
#     参数：
#     - X_world: np.array, 形状 (3,), 3D点在世界坐标
#     - uv_obs: np.array, 形状 (2,), 对应观测像素坐标
#     - R_w2c: np.array, (3,3), 世界到相机旋转矩阵
#     - t_w2c: np.array, (3,), 世界到相机平移向量
#     - K: np.array, (3,3), 相机内参矩阵
#     返回：
#     - error: float, 像素级重投影误差
#     """
#     # 世界坐标 -> 相机坐标
#     X_cam = R_w2c @ X_world + t_w2c  # shape (3,)
#     # 相机坐标 -> 像素坐标
#     x_homog = K @ X_cam  # shape (3,)
#     uv_proj = x_homog[:2] / x_homog[2]  # 齐次坐标归一化
#     # 计算像素距离（欧氏距离）
#     error = np.linalg.norm(uv_proj - uv_obs)
#     return error

# reproj_errors = []
# for i, cam in enumerate(cameras_optimized):
#     R = np.array(cam["R_w2c"])
#     t = np.array(cam["t_w2c"])
#     cam_id = cam["id"]
#     # 找到该相机位姿下的所有2d点
#     cam_observations = [obs for obs in observations if obs["cam_id"] == cam_id]
#     print(f"{len(cam_observations)} points_2d in camera {cam_id}")
#     for j, init_point in enumerate(cam_observations):
#         pt_id = init_point["pt_id"]
#         pt_2d = init_point["uv"]
#         pts_3d = [pt for pt in points_optimized if pt["id"] == pt_id]
#         pt_3d = np.array(pts_3d[0]['X'])
#         reproj_error = reprojection_error(pt_3d, pt_2d, R, t, K)
#         reproj_errors.append(reproj_error)
#         print(f"camera_id: {i}, point_id: {j}, re: {reproj_error}")
    

# reproj_error_mean = np.mean(reproj_errors)
# print("Mean Reprojection Error (px):", reproj_error_mean)

# -----------------------------
# 3. 点云统计信息
# -----------------------------
# def pointcloud_stats(points):
#     N = points.shape[0]
#     min_bound = points.min(axis=0)
#     max_bound = points.max(axis=0)
#     volume = np.prod(max_bound - min_bound)
#     density = N / volume
#     return N, volume, density

# # 稀疏点云
# sparse_N, sparse_vol, sparse_density = pointcloud_stats(points_sparse)

# # 稠密点云
# pcd_dense = o3d.io.read_point_cloud(dense_ply_path)
# points_dense = np.asarray(pcd_dense.points)
# dense_N, dense_vol, dense_density = pointcloud_stats(points_dense)

# print(f"Sparse: {sparse_N} points, volume={sparse_vol:.2f}, density={sparse_density:.2f}")
# print(f"Dense: {dense_N} points, volume={dense_vol:.2f}, density={dense_density:.2f}")

# # -----------------------------
# # 4. 执行时间（手动输入或记录）
# # -----------------------------
# # 这里可根据日志或计时记录每个阶段耗时
# runtime_dict = {
#     "sparse_reconstruction_sec": 120,  # 示例
#     "BA_optimization_sec": 300,
#     "dense_reconstruction_sec": 600
# }

# # -----------------------------
# # 5. 保存结果表格
# # -----------------------------
# metrics = {
#     "Metric": ["ReprojectionError(px)", "SparsePoints", "SparseVolume", "SparseDensity",
#                "DensePoints", "DenseVolume", "DenseDensity",
#                "SparseRuntime(s)", "BA_Runtime(s)", "DenseRuntime(s)"],
#     "Value": [reproj_error_mean, sparse_N, sparse_vol, sparse_density,
#               dense_N, dense_vol, dense_density,
#               runtime_dict["sparse_reconstruction_sec"],
#               runtime_dict["BA_optimization_sec"],
#               runtime_dict["dense_reconstruction_sec"]]
# }

# df = pd.DataFrame(metrics)
# df.to_csv(output_csv, index=False)
# print(f"Metrics saved to {output_csv}")
