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
    uv_proj = uv_proj_h[:, :2] / (uv_proj_h[:, 2:3] + 1e-10)  # Added epsilon to prevent divide by zero
    errors = np.linalg.norm(uv_proj - uvs_obs, axis=1)
    return errors, uv_proj

def point_cloud_density(points, auto_detect_units=True):
    """
    Calculate point cloud density with automatic unit detection and conversion
    This is the FIXED version that handles coordinate units correctly.
    
    Args:
        points: (N,3) array of 3D points
        auto_detect_units: If True, automatically detect and convert units
    
    Returns:
        density: density in points per cubic meter (m³)
        info: dict with unit information
    """
    min_pt = points.min(axis=0)
    max_pt = points.max(axis=0)
    bbox_size = max_pt - min_pt
    max_dim = bbox_size.max()
    
    if auto_detect_units:
        # Detect unit system based on bounding box size
        if max_dim > 1000:
            unit = "millimeters"
            scale = 1000.0
        elif max_dim > 100:
            unit = "centimeters"
            scale = 100.0
        else:
            unit = "meters"
            scale = 1.0
    else:
        unit = "unknown"
        scale = 1.0
    
    # Calculate volume in meters³
    volume_m3 = np.prod(bbox_size / scale)
    density_per_m3 = len(points) / volume_m3 if volume_m3 > 0 else 0
    
    info = {
        'unit': unit,
        'scale': scale,
        'bbox_size_original': bbox_size,
        'bbox_size_meters': bbox_size / scale,
        'volume_m3': volume_m3,
        'points': len(points)
    }
    
    return density_per_m3, info

# OLD INCORRECT VERSION (for reference):
# def point_cloud_density(points):
#     # 点云边界盒体积
#     min_pt = points.min(axis=0)
#     max_pt = points.max(axis=0)
#     volume = np.prod(max_pt - min_pt)  # ❌ No unit conversion!
#     density = len(points) / volume
#     return density  # ❌ Returns density in original units, not m³

def point_distribution_uniformity(points, voxel_size=0.1):
    # 将点划分到体素
    voxel_coords = np.floor(points / voxel_size).astype(int)
    # 每个体素点数
    unique, counts = np.unique(voxel_coords, axis=0, return_counts=True)
    variance = np.var(counts)
    return variance

def project_points_to_image(points_3d, R, t, K, image, z_min=0.1, z_max=100):
    """
    project 3D points to the input image and filter the abnormal points
    points_3d: (N,3) ndarray
    R: (3,3) camera rotation matrix
    t: (3,) camera translation matrix
    K: (3,3) camera intrinsic parameters
    image: original image ndarray
    z_min, z_max: depth range
    """
    img_proj = image.copy()
    h, w = img_proj.shape[:2]

    for P_world in points_3d:
        # 世界坐标 -> 相机坐标
        P_cam = R @ P_world + t

        # 过滤在相机背面的点和异常深度
        if P_cam[2] <= z_min or P_cam[2] >= z_max:
            continue

        # 相机坐标 -> 像素坐标
        p = K @ (P_cam / P_cam[2])
        u, v = int(round(p[0])), int(round(p[1]))

        # 判断是否在图像内
        if 0 <= u < w and 0 <= v < h:
            cv2.circle(img_proj, (u, v), radius=5, color=(0,0,255), thickness=-1)

    return img_proj


# scenes = ["arch", "chinese_heritage_centre", "pavilion"]
scenes = ["arch", "chinese_heritage_centre"]
for scene in scenes:
    ba_initial_path = f"project/3D_Reconstruction_Pipeline/new_result/{scene}/non_sequential/ba_problem_export.json"
    ba_refined_path = f"project/3D_Reconstruction_Pipeline/new_result/{scene}/non_sequential/ba_problem_ceres_refined.json"
    sparse_points_path = f"project/3D_Reconstruction_Pipeline/new_result/{scene}/non_sequential/initial/sparse_points_initial.ply"
    sparse_points3d_path = f"project/3D_Reconstruction_Pipeline/result/{scene}/colmap_model_refined/points3D.txt"
    dense_points_path = f"project/3D_Reconstruction_Pipeline/result/{scene}/dense_model/dense.ply"
    save_dir = f'project/evaluation/new_results_test/{scene}'
    os.makedirs(save_dir, exist_ok=True)
    # -----------------------------
    # 1. Load data
    # -----------------------------
    with open(ba_initial_path, 'r') as f:
        init_data = json.load(f)

    with open(ba_refined_path, 'r') as f:
        refined_data = json.load(f)

    K = np.array(refined_data["K"])
    cameras_optimized = refined_data["cameras_optimized"]
    points_optimized = refined_data["points_optimized"]
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
        
        cam_obs = obs_by_cam[cam_id]
        if len(cam_obs) == 0:
            continue
        
        # prepare for calculation by batch
        X_worlds = np.array([points_dict[obs['pt_id']] for obs in cam_obs])
        if len(X_worlds) > max_num_ponts:
            max_num_ponts = len(X_worlds)
            max_cam_id = cam_id
            max_cam_index = i
        uvs_obs = np.array([obs['uv'] for obs in cam_obs])
        
        errors, uv_proj = reprojection_error_batch(X_worlds, uvs_obs, R, t, K)
        # Filter out invalid errors
        valid_errors = errors[np.isfinite(errors)]
        all_errors.extend(valid_errors.tolist())
    
    # read original image with the most 3D points
    R_max = np.array(cameras_optimized[max_cam_index]["R_w2c"])
    t_max = np.array(cameras_optimized[max_cam_index]["t_w2c"])
    frame_path = f"project/scenes/images/{scene}/{frame_by_cam[max_cam_id]}.jpg"
    img = cv2.imread(frame_path)
    if img is not None:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        points_3d = [np.array(point['X']) for point in points_optimized]
        img_proj = project_points_to_image(points_3d, R_max, t_max, K, img_rgb)
        cv2.imwrite(f'{save_dir}/reprojection_error_heatmap_{frame_by_cam[max_cam_id]}.png', 
                cv2.cvtColor(img_proj, cv2.COLOR_RGB2BGR))

    # -----------------------------
    # 3. Calculate points density and point distribution uniformity
    # -----------------------------
    # Sparse point cloud - use BA optimized version (points3D.txt) if available, otherwise use initial
    sparse_density = None
    sparse_info = None
    sparse_points = None
    
    # Try to load from BA optimized points3D.txt first (this is the correct sparse point cloud)
    if os.path.exists(sparse_points3d_path):
        try:
            # Read COLMAP points3D.txt format: POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]
            points_list = []
            with open(sparse_points3d_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 4:
                            try:
                                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                                points_list.append([x, y, z])
                            except (ValueError, IndexError):
                                continue
            if points_list:
                sparse_points = np.array(points_list)
                sparse_density, sparse_info = point_cloud_density(sparse_points)
                print(f"✅ Loaded sparse point cloud from BA optimized points3D.txt: {len(sparse_points)} points")
        except Exception as e:
            print(f"Warning: Could not read points3D.txt: {e}")
    
    # Fallback to initial sparse point cloud if points3D.txt not available
    if sparse_points is None and os.path.exists(sparse_points_path):
        try:
            sparse_ply = PlyData.read(sparse_points_path)
            sparse_vertex = sparse_ply['vertex']
            sparse_points = np.column_stack([sparse_vertex['x'], sparse_vertex['y'], sparse_vertex['z']])
            sparse_density, sparse_info = point_cloud_density(sparse_points)
            print(f"⚠️  Using initial sparse point cloud (not BA optimized): {len(sparse_points)} points")
        except Exception as e:
            print(f"Warning: Could not analyze sparse point cloud: {e}")
    
    # Dense point cloud (this is what matters for final quality)
    dense_density = None
    dense_info = None
    if os.path.exists(dense_points_path):
        try:
            dense_ply = PlyData.read(dense_points_path)
            dense_vertex = dense_ply['vertex']
            dense_points = np.column_stack([dense_vertex['x'], dense_vertex['y'], dense_vertex['z']])
            dense_density, dense_info = point_cloud_density(dense_points)
        except Exception as e:
            print(f"Warning: Could not analyze dense point cloud: {e}")
    
    # Distribution uniformity (using sparse point cloud)
    if sparse_points is not None:
        try:
            distribution_uniformity = point_distribution_uniformity(sparse_points)
        except:
            distribution_uniformity = 0.0
    else:
        distribution_uniformity = 0.0

    # -----------------------------
    # 4. Calculate total mean error and plot figure
    # -----------------------------
    all_errors = np.array(all_errors)
    # plot histogram
    plt.figure(figsize=(8,5))
    plt.hist(all_errors, bins=100, color='skyblue', edgecolor='black')
    plt.title('Reprojection Error Distribution')
    plt.xlabel('Reprojection Error (pixels)')
    plt.ylabel('Number of Points')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(f'{save_dir}/reprojection_error_hist.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/reprojection_error_hist.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    output_file = f'{save_dir}/metrics.txt'
    with open(output_file, 'w') as f:
        # Reprojection errors
        mean_str = f"{scene} Overall mean reprojection error: {all_errors.mean():.3f} px\n"
        print(mean_str, end='')
        f.write(mean_str)

        median_str = f"{scene} Overall median reprojection error: {np.median(all_errors):.3f} px\n"
        print(median_str, end='')
        f.write(median_str)

        max_str = f"{scene} Overall max reprojection error: {all_errors.max():.3f} px\n"
        print(max_str, end='')
        f.write(max_str)
        
        # Sparse point cloud density (BA optimized - this is what matters for standard 1-50 points/m³)
        if sparse_density is not None:
            sparse_str = f"{scene} Sparse point cloud density: {sparse_density:.4f} points/m³ (coordinate unit: {sparse_info['unit']})\n"
            print(sparse_str, end='')
            f.write(sparse_str)
            
            # Assessment against standard (1-50 points/m³)
            if 1 <= sparse_density <= 50:
                f.write(f"  ✅ Sparse point cloud meets standard (1-50 points/m³)\n")
            elif sparse_density > 50:
                f.write(f"  ✅ Sparse point cloud EXCEEDS standard (very dense!)\n")
            else:
                f.write(f"  ⚠️  Sparse point cloud below standard (<1 points/m³) - may need more images or better feature matching\n")
        
        # Dense point cloud density (this is what matters!)
        if dense_density is not None:
            dense_str = f"{scene} Dense point cloud density: {dense_density:.2f} points/m³ (coordinate unit: {dense_info['unit']})\n"
            print(dense_str, end='')
            f.write(dense_str)
            
            # Assessment
            if 1 <= dense_density <= 50:
                f.write(f"  ✅ Dense point cloud meets standard (1-50 points/m³)\n")
            elif dense_density > 50:
                f.write(f"  ✅ Dense point cloud EXCEEDS standard (very dense!)\n")
            else:
                f.write(f"  ⚠️  Dense point cloud below standard (<1 points/m³)\n")
        
        # Distribution uniformity
        uniformity_str = f"{scene} Point Distribution Uniformity: {distribution_uniformity:.4f}\n"
        print(uniformity_str, end='')
        f.write(uniformity_str)

