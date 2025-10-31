import os
import numpy as np

CALIB_PATH = "../calibration/results/iphone_calibration.npz"
POSES_DIR  = "../feature_matching/result/arch/poses"
MATCH_DIR  = "../feature_matching/result/arch/matches"
SUMMARY_CSV = "../feature_matching/result/arch_pose_summary.csv"
C_OUTPUT_DIR = "result/arch"
IMAGE_DIR = "../scenes/images/arch"
ALIGN_WITH_PCA = True
NORMALIZE_SCALE = True
os.makedirs(C_OUTPUT_DIR, exist_ok=True)

VOXEL_SIZE = 0.01
REPROJ_THRESH_PX = 3.5
MIN_POINTS_PER_PAIR_FOR_MERGE = 30

def load_calibration():

    K = np.array([[3183.592270336112,     0.0, 1911.3122172243477],
                  [   0.0,              3183.8330165146126, 1100.814254448177],
                  [   0.0,                 0.0,               1.0]], dtype=np.float64)
    dist = np.zeros((5,1), dtype=np.float64)
    return K, dist
#def load_calibration():
 #   data = np.load(CALIB_PATH, allow_pickle=True)
  #  K = data["K"].astype(np.float64)
   # dist = np.zeros((5,1), dtype=np.float64)
    #return K, dist
