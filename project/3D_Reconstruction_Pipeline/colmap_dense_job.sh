#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=pmstereo
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
echo "[DEBUG] Listing sparse/images structure under dense_model:"
ls -R result/arch/dense_model/sparse
ls -R result/arch/dense_model/images

echo "[INFO] Running COLMAP PatchMatchStereo on GPU..."
colmap patch_match_stereo \
    --workspace_path result/arch/dense_model \
    --workspace_format COLMAP \
    --PatchMatchStereo.max_image_size 2000 \
    --PatchMatchStereo.geom_consistency true

echo "Job finished at $(date)"
