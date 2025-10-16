import glob
import os
import numpy as np
import cv2 as cv

# ======== 1) 基本設定 ========
# 你的棋盤內角點數 (注意：是內角點！例如棋盤印 10x7 格，內角點就可能是 9x6)
CHESSBOARD_SIZE = (9, 6)        # (cols, rows) 內角點數
SQUARE_SIZE = 1.0               # 每一格實際邊長（任意單位；只影響外部尺度，標定K和畸變不受此值影響）
IMAGE_DIR = "/Users/chongtszkwan/Documents/side-projects/CV_Assignment1/project/calibration/frames"            # 棋盤影像資料夾
IMAGE_PATTERN = os.path.join(IMAGE_DIR, "frame_*.jpg")  # 依你的檔名調整

# 角點細化參數
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

# ======== 2) 準備 3D-2D 對應關係 ========
# 以棋盤平面 Z=0 建立單格座標（乘上 SQUARE_SIZE 得到實際尺寸）
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []  # 3D：多張影像共用的棋盤3D點
imgpoints = []  # 2D：每張影像偵測到的角點像素座標
img_size = None

# ======== 3) 讀圖 & 偵測棋盤角點 ========
images = sorted(glob.glob(IMAGE_PATTERN))
print(f"[INFO] 找到 {len(images)} 張影像")
if len(images) == 0:
    raise SystemExit("找不到影像，請確認 IMAGE_DIR / IMAGE_PATTERN 設定。")

valid_count = 0
for fname in images:
    img = cv.imread(fname)
    if img is None:
        print(f"[WARN] 無法讀取：{fname}")
        continue
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    if img_size is None:
        img_size = (gray.shape[1], gray.shape[0])  # (width, height)

    # 偵測內角點
    ret, corners = cv.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
    if ret:
        # 亞像素角點優化
        corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners)
        valid_count += 1

        # 視覺化（可選）：把偵測到的角點畫在圖上並存檔
        vis = img.copy()
        cv.drawChessboardCorners(vis, CHESSBOARD_SIZE, corners, ret)
        out_path = os.path.join(IMAGE_DIR, f"detect_{os.path.basename(fname)}")
        cv.imwrite(out_path, vis)
    else:
        print(f"[WARN] 未偵測到角點：{fname}")

print(f"[INFO] 成功偵測角點的影像數：{valid_count}")
if valid_count < 8:
    print("[WARN] 成功影像少於 8，標定可能不穩定，建議多補拍一些不同角度的棋盤影像。")

# ======== 4) 相機標定 ========
print("[INFO] 開始 calibrateCamera() ...")
ret, K, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, img_size, None, None)

print("\n=== 標定結果 ===")
print("RMS 重投影誤差 (ret):", ret)
print("內參矩陣 K:\n", K)
print("畸變係數 dist (k1,k2,p1,p2[,k3[,k4,k5,k6]]):\n", dist.ravel())

# ======== 5) 計算平均重投影誤差（逐張） ========
total_error = 0.0
total_points = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
    err = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)
    total_error += err**2
    total_points += len(imgpoints2)
mean_error = np.sqrt(total_error / total_points)
print("平均重投影RMSE（每點）:", mean_error)

# ======== 6) 存檔 ========
results_dir = "/Users/chongtszkwan/Documents/side-projects/CV_Assignment1/project/calibration/results"
os.makedirs(results_dir, exist_ok=True)
np.savez(os.path.join(results_dir, "iphone_calibration.npz"), K=K, dist=dist, rms=ret, mean_rmse=mean_error, img_size=img_size)
print("\n[INFO] 已儲存到 results/iphone_calibration.npz")

# ======== 7) 示範：對一張圖做 undistort（可選） ========
sample = cv.imread(images[0])
h, w = sample.shape[:2]
newK, roi = cv.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0)  # alpha=0裁切、=1保留全視野
undist = cv.undistort(sample, K, dist, None, newK)
cv.imwrite(os.path.join(results_dir, "undistort_sample.jpg"), undist)
print("[INFO] 已輸出範例去畸變影像：results/undistort_sample.jpg")
