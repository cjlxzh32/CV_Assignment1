# 3D_Reconstruction_Pipeline

## step 1 稀疏点云重建

python run_stage1_sparse.py
得到结果在result/arch/initial里
poses_initial.json是全局相机位姿
sparse_metrics.csv是每对图像的匹配质量统计
sparse_points_initial.ply是稀疏点云

## step 2 采用Ceres做高质量全局BA优化（这一部分是下载开源Ceres，代码没有上传GitHub）

python export_to_ceres.py
得到结果在result/arch里，把稀疏重建问题整理成标准 BA 格式：相机、3D点、观测的整体图
ba_problem_export.json
ba_problem_export.npz

下一阶段是在 C++ 里完成的，需要先编译 ceres_ba_runner。编译过程已经完成并通过（自建无-SuiteSparse版本的 Ceres）
cd 3D_Reconstruction_Pipeline/ceres_ba/build

./ceres_ba_runner \
    --input ../../result/arch/ba_problem_export.json \
    --output ../../result/arch/ba_problem_ceres_refined.json \
    --fix_first_camera 1 \
    --huber_delta 3.0
优化完成后输出到result/arch/ba_problem_ceres_refined.json
cameras_optimized: 全局一致的最终相机位姿
points_optimized: 全局一致、降噪后的稀疏点云
notes: BA 优化信息

## step 3 密集点云重建

cd 3D_Reconstruction_Pipeline
python export_refined_to_colmap.py
得到result/arch/colmap_model_refined/
    cameras.txt
    images.txt
    points3D.txt

## 1. undistort

colmap image_undistorter \
    --image_path ../scenes/images/arch \
    --input_path result/arch/colmap_model_refined \
    --output_path result/arch/dense_model \
    --output_type COLMAP \
    --max_image_size 2000

## 2. PatchMatch stereo

sbatch colmap_dense_job.sh
cat logs/error_pmstereo_*.err
squeue -u haorong001
colmap patch_match_stereo \
    --workspace_path result/arch/dense_model \
    --workspace_format COLMAP \
    --PatchMatchStereo.max_image_size 2000 \
    --PatchMatchStereo.geom_consistency true

## 3. depth fusion

sbatch colmap_fusion_job.sh
cat logs/error_stereo_fusion_*.err
colmap stereo_fusion \
    --workspace_path result/arch/dense_model \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path result/arch/dense_model/dense.ply
最后得到result/arch/dense_model/dense.ply最终稠密点云
python post_clean_dense.py得到干净点云
