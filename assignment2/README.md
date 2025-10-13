# Stereo Disparity Map Generator

A Python script that computes disparity maps from stereo image pairs using SSD-based block matching.

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Matplotlib

## Usage

```bash
python3 disparity_map.py
```

## Output

The script processes stereo image pairs and generates disparity maps in the `maps/` folder:

- **Original grayscale**: `disparity_[image_name].png`
- **Colored (JET)**: `disparity_colored_[image_name].png`
- **Multiple colormaps**: `disparity_[image_name]_[colormap].png`

Available colormaps: `jet`, `hot`, `viridis`, `plasma`, `inferno`

## Input Images

Place stereo image pairs in the `img/` folder:
- `corridorl.jpg` & `corridorr.jpg`
- `triclopsi2l.jpg` & `triclopsi2r.jpg`

## Parameters

- **Block size**: 5x5 pixels
- **Max disparity**: 64 pixels
- **Algorithm**: Sum of Squared Differences (SSD)

## File Structure

```
assignment2/
├── disparity_map.py          # Main script
├── img/                      # Input stereo images
│   ├── corridorl.jpg
│   ├── corridorr.jpg
│   ├── triclopsi2l.jpg
│   └── triclopsi2r.jpg
└── maps/                     # Generated disparity maps
    ├── disparity_corridorl.png
    ├── disparity_colored_corridorl.png
    ├── disparity_triclopsi2l.png
    └── disparity_colored_triclopsi2l.png
```
