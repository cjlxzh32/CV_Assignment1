Features extracted for all images in /Users/chongtszkwan/Documents/side-projects/CV_Assignment1/project/scenes/images/chinese_heritage_centre
- Detector: SIFT
- Calibration used: Yes
- Summary CSV: feature_summary.csv
- Keypoints: *_keypoints.npy (columns: x, y, size, angle, response, octave, class_id)
- Descriptors: *_descriptors.npy (shape: [N, D])
Consumers: load matching image pairs, match descriptors (FLANN/BF),
then use K (from calibration) for recoverPose/triangulate.
