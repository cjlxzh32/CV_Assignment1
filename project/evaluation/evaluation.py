import json, os
import numpy as np
from plyfile import PlyData
from collections import defaultdict
import matplotlib.pyplot as plt
import cv2


def reprojection_error_batch(X_worlds, uvs_obs, R_w2c, t_w2c, K):
    """
    Calculate reprojection error by batch
    X_worlds: (N,3)
    uvs_obs: (N,2)
    R_w2c: (3,3)
    t_w2c: (3,)
    K: (3,3)
    return: reprojection error of each point (N,)
    """
    X_cam = (R_w2c @ X_worlds.T + t_w2c[:, None]).T # (N,3)
    uv_proj_h = (K @ X_cam.T).T # (N,3)
    uv_proj = uv_proj_h[:, :2] / uv_proj_h[:, 2:3]
    errors = np.linalg.norm(uv_proj - uvs_obs, axis=1)
    return errors, uv_proj

def point_cloud_density(points):
    # 点云边界盒体积
    min_pt = points.min(axis=0)
    max_pt = points.max(axis=0)
    volume = np.prod(max_pt - min_pt)
    
    density = len(points) / volume  # 点数 / 体积
    return density

def point_distribution_uniformity(points, voxel_size=0.1):
    # 将点划分到体素
    voxel_coords = np.floor(points / voxel_size).astype(int)
    # 每个体素点数
    unique, counts = np.unique(voxel_coords, axis=0, return_counts=True)
    variance = np.var(counts)
    return variance

# def project_points_to_image(points_3d, R, t, K, image):
#     """
#     将3D点投影到输入图像上
#     points_3d: (N,3) ndarray
#     R: (3,3) 相机旋转矩阵
#     t: (3,) 相机平移向量
#     K: (3,3) 相机内参
#     image: 原始图像 ndarray
#     """
#     img_proj = image.copy()
#     h, w = img_proj.shape[:2]

#     for P_world in points_3d:
#         # 世界坐标 -> 相机坐标
#         P_cam = R @ P_world + t
#         if P_cam[2] <= 0:
#             continue  # 点在相机背面

#         # 相机坐标 -> 像素坐标
#         p = K @ (P_cam / P_cam[2])
#         u, v = int(round(p[0])), int(round(p[1]))

#         # 判断是否在图像内
#         if 0 <= u < w and 0 <= v < h:
#             cv2.circle(img_proj, (u, v), radius=5, color=(0,0,255), thickness=-1)

#     return img_proj

def project_points_to_image(points_3d, R, t, K, image, z_min=0.1, z_max=100):
    """
    将3D点投影到输入图像上，并过滤异常点
    points_3d: (N,3) ndarray
    R: (3,3) 相机旋转矩阵
    t: (3,) 相机平移向量
    K: (3,3) 相机内参
    image: 原始图像 ndarray
    z_min, z_max: 深度范围，过滤掉离相机太近或太远的点
    """
    img_proj = image.copy()
    h, w = img_proj.shape[:2]

    for P_world in points_3d:
        # 世界坐标 -> 相机坐标
        P_cam = R @ P_world + t

        # 过滤在相机背面的点和异常深度
        # if P_cam[2] <= z_min or P_cam[2] > z_max:
        #     continue
        if P_cam[2] <= z_min:
            continue

        # 相机坐标 -> 像素坐标
        p = K @ (P_cam / P_cam[2])
        u, v = int(round(p[0])), int(round(p[1]))

        # 判断是否在图像内
        if 0 <= u < w and 0 <= v < h:
            cv2.circle(img_proj, (u, v), radius=5, color=(0,0,255), thickness=-1)

    return img_proj


scenes = ["arch", "chinese_heritage_centre", "pavilion"]
for scene in scenes:
    ba_initial_path = f"project/3D_Reconstruction_Pipeline/result/{scene}/ba_problem_export.json"
    ba_refined_path = f"project/3D_Reconstruction_Pipeline/result/{scene}/ba_problem_ceres_refined_A.json"
    sparse_points_path = f"project/3D_Reconstruction_Pipeline/result/{scene}/initial/sparse_points_initial.ply"
    os.makedirs(f'project/evaluation/results/{scene}', exist_ok=True)
    # -----------------------------
    # 1. Load data
    # -----------------------------
    with open(ba_initial_path, 'r') as f:
        init_data = json.load(f)

    with open(ba_refined_path, 'r') as f:
        refined_data = json.load(f)

    K = np.array(refined_data["K"])
    cameras_optimized = refined_data["cameras_optimized"]
    points_optimized = refined_data["points_optimized"] # 16647
    cameras_init = init_data["cameras"]
    observations = init_data["observations"]

    # construct pt_id -> 3D point dictionary
    points_dict = {pt['id']: np.array(pt['X']) for pt in points_optimized}

    # construct cam_id -> observation list dictionary
    obs_by_cam = defaultdict(list)
    for obs in observations:
        obs_by_cam[obs['cam_id']].append(obs)
        
    # construct cam_id -> frame dictionary
    frame_by_cam = {}
    for cam in cameras_init:
        frame_by_cam[cam["id"]] = cam["name"]

    # -----------------------------
    # 2. Calculate reprojection error by batch
    # -----------------------------
    all_errors = []
    max_cam_id = -1
    max_num_ponts = 0
    max_cam_index = -1
    for i, cam in enumerate(cameras_optimized):
        cam_id = cam["id"]
        R = np.array(cam["R_w2c"])
        t = np.array(cam["t_w2c"])
        # frame_path = f"project/scenes/images/{scene}/{frame_by_cam[cam_id]}.jpg"
        
        cam_obs = obs_by_cam[cam_id]
        if len(cam_obs) == 0:
            continue
        
        # prepare for calculation by batch
        X_worlds = np.array([points_dict[obs['pt_id']] for obs in cam_obs]) # 3D point
        if len(X_worlds) > max_num_ponts:
            max_num_ponts = len(X_worlds)
            max_cam_id = cam_id
            max_cam_index = i
        uvs_obs = np.array([obs['uv'] for obs in cam_obs]) # 2D point
        
        errors, uv_proj = reprojection_error_batch(X_worlds, uvs_obs, R, t, K)
        all_errors.extend(errors)
        
        # print(f"Camera {cam_id}: {len(errors)} points, mean reprojection error = {errors.mean():.3f} px")
    
    # read original image with the most 3D points
    R_max = np.array(cameras_optimized[max_cam_index]["R_w2c"])
    t_max = np.array(cameras_optimized[max_cam_index]["t_w2c"])
    frame_path = f"project/scenes/images/{scene}/{frame_by_cam[max_cam_id]}.jpg"
    img = cv2.imread(frame_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    points_3d = [np.array(point['X']) for point in points_optimized]
    img_proj = project_points_to_image(points_3d, R_max, t_max, K, img_rgb)
    cv2.imwrite(f'project/evaluation/results/{scene}/reprojection_error_heatmap_{frame_by_cam[max_cam_id]}.png', 
            cv2.cvtColor(img_proj, cv2.COLOR_RGB2BGR))

    # -----------------------------
    # 3. Calculate points density and point distribution uniformity
    # -----------------------------
    points = np.vstack([PlyData.read(sparse_points_path)['vertex'][axis] for axis in ['x','y','z']]).T # 23109
    density = point_cloud_density(points)
    distribution_uniformity = point_distribution_uniformity(points)

    # -----------------------------
    # 4. Calculate total mean error and plot figure
    # -----------------------------
    all_errors = np.array(all_errors)
    # plt.hist(all_errors, bins=50, color='skyblue', edgecolor='black')
    # plot histogram
    plt.figure(figsize=(8,5))
    plt.hist(all_errors, bins=100, color='skyblue', edgecolor='black')
    plt.title('Reprojection Error Distribution')
    plt.xlabel('Reprojection Error (pixels)')
    plt.ylabel('Number of Points')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(f'project/evaluation/results/{scene}/reprojection_error_hist.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    output_file = f'project/evaluation/results/{scene}/metrics.txt'
    with open(output_file, 'w') as f:
        # write mean
        mean_str = f"{scene} Overall mean reprojection error: {all_errors.mean():.3f} px\n"
        print(mean_str, end='')
        f.write(mean_str)

        # write median
        median_str = f"{scene} Overall median reprojection error: {np.median(all_errors):.3f} px\n"
        print(median_str, end='')
        f.write(median_str)

        # write max
        max_str = f"{scene} Overall max reprojection error: {all_errors.max():.3f} px\n"
        print(max_str, end='')
        f.write(max_str)
        
        # write point cloud density
        mean_str = f"{scene} Point cloud density: {density:.4f} points/unit volume\n"
        print(mean_str, end='')
        f.write(mean_str)

        # write point distribution uniformity
        median_str = f"{scene} Point Distribution Uniformity: {distribution_uniformity:.4f}\n"
        print(median_str, end='')
        f.write(median_str)
