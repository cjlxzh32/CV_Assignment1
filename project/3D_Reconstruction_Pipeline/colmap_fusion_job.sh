#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=stereo_fusion
#SBATCH --output=logs/output_%x_%j.out
#SBATCH --error=logs/error_%x_%j.err


mkdir -p logs

echo "Job started on $(hostname) at $(date)"

module load cuda/12.8.0
module load anaconda
eval "$(conda shell.bash hook)"
conda activate zhr


cd /home/msai/haorong001/CV_Assignment1/project/3D_Reconstruction_Pipeline

echo "[DEBUG] Current working dir: $(pwd)"

echo "[DEBUG] Checking that PatchMatchStereo outputs exist:"
echo "[DEBUG] depth_maps sample:"
ls -l result/arch/dense_model/stereo/depth_maps | head
echo "[DEBUG] normal_maps sample:"
ls -l result/arch/dense_model/stereo/normal_maps | head

echo "[INFO] Running COLMAP stereo_fusion..."
colmap stereo_fusion \
    --workspace_path result/arch/dense_model \
    --workspace_format COLMAP \
    --input_type geometric \
    --StereoFusion.max_reproj_error 4 \
    --StereoFusion.max_depth_error 0.02 \
    --StereoFusion.min_num_pixels 3 \
    --StereoFusion.max_normal_error 30 \
    --StereoFusion.check_num_images 30 \
    --output_path result/arch/dense_model/dense.ply
   

echo "[INFO] stereo_fusion completed."
echo "[INFO] Output point cloud: result/arch/dense_model/dense.ply"
echo "Job finished at $(date)"
