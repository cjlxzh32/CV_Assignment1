
Video to photos
ffmpeg -i camera.MOV -vf fps=2 calibration_frames/frame_%03d.jpg


Camera Calibration Results:
- K and dist saved in results/iphone_calibration.npz
- All 60 frames used; RMS error = 2.06 px

Feature Extraction:
- Extracted using SIFT (OpenCV)
- Each image’s keypoints & descriptors saved in /results/features/
- Keypoints saved as .npy; descriptors as .npy; matching to be done by Group B