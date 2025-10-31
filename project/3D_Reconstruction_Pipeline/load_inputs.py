import numpy as np
import csv
from pathlib import Path
from typing import Dict, Tuple, List
from config_paths import SUMMARY_CSV, MATCH_DIR, POSES_DIR

def load_summary_pairs():

    pairs = []
    meta_rows = []
    with open(SUMMARY_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pair_name = row["pair"]
            imgA, imgB = pair_name.split("-")
            pairs.append((imgA, imgB))
            meta_rows.append(row)
    return pairs, meta_rows

def load_pair_inliers(imgA: str, imgB: str):

    npz_path = Path(MATCH_DIR) / f"{imgA}_{imgB}_inliers.npz"
    data = np.load(npz_path)
    ptsA = data["pts1"].astype(np.float64)
    ptsB = data["pts2"].astype(np.float64)
    return ptsA, ptsB

def load_pair_pose(imgA: str, imgB: str):

    txt_path = Path(POSES_DIR) / f"{imgA}_{imgB}.txt"
    vals = np.loadtxt(txt_path).astype(np.float64)
    if vals.ndim > 1:
        vals = vals[0]
    R = vals[:9].reshape(3,3)
    t = vals[9:].reshape(3,1)
    return R, t