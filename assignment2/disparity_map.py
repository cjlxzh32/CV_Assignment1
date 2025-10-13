import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def compute_disparity(left_img, right_img, block_size=5, max_disparity=64):
    """
    Compute disparity map between two stereo images using SSD-based block matching.
    """
    # RGB to grayscale
    if len(left_img.shape) > 2:
        left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    if len(right_img.shape) > 2:
        right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    
    # img dimensions
    height, width = left_img.shape
    
    # init disparity map
    disparity_map = np.zeros((height, width), dtype=np.float32)
    
    # Half block size for window extraction
    half_block = block_size // 2
    
    # for every valid pixel (ignore borders)
    for y in range(half_block, height - half_block):
        for x in range(half_block, width - half_block - max_disparity):
            # Extract block from left image
            left_block = left_img[y - half_block:y + half_block + 1, 
                                  x - half_block:x + half_block + 1]
            
            # init best match
            min_cost = float('inf')
            best_disparity = 0
            
            # search possible disparities
            for d in range(max_disparity + 1):
                # Extract corresponding block from right image (offset left by d)
                right_block = right_img[y - half_block:y + half_block + 1, 
                                         x - half_block - d:x + half_block + 1 - d]
                
                # skip if out of range
                if right_block.shape != left_block.shape:
                    continue
                
                # compute SSD cost
                ssd_cost = np.sum((left_block - right_block) ** 2)
                
                # keep disparity with the lowest cost
                if ssd_cost < min_cost:
                    min_cost = ssd_cost
                    best_disparity = d
            
            # save the best disparity value for this pixel
            disparity_map[y, x] = best_disparity
    
    # normalize disparity map to 0–2555)
    if np.max(disparity_map) > 0:
        disparity_map = (disparity_map / np.max(disparity_map) * 255).astype(np.uint8)
    else:
        disparity_map = disparity_map.astype(np.uint8)
    
    return disparity_map

def apply_colormap(disparity_map, colormap='jet'):
    """
    Apply a color map to make the disparity map easier to interpret visually.
    """
    normalized = disparity_map.astype(np.float32)
    if np.max(normalized) > 0:
        normalized = normalized / np.max(normalized)
    
    # color mapping
    if colormap == 'jet':
        colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    elif colormap == 'hot':
        colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_HOT)
    elif colormap == 'viridis':
        colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    elif colormap == 'plasma':
        colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_PLASMA)
    elif colormap == 'inferno':
        colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    else:
        colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    return colored

def save_colormap_comparison(disparity_map, filename_prefix='disparity'):
    """
    Save disparity maps with multiple color mappings for comparison.
    """
    colormaps = ['jet', 'hot', 'viridis', 'plasma', 'inferno']
    
    for cmap in colormaps:
        colored = apply_colormap(disparity_map, cmap)
        filename = f'{filename_prefix}_{cmap}.png'
        cv2.imwrite(filename, colored)
        print(f"Colored disparity map ({cmap}) saved as '{filename}'")

def process_stereo_pair(left_path, right_path, block_size=5, max_disparity=64):
    """
    Process a stereo image pair and generate disparity maps.
    """
    print(f"\nProcessing stereo pair: {os.path.basename(left_path)} & {os.path.basename(right_path)}")
    
    left_image = cv2.imread(left_path)
    right_image = cv2.imread(right_path)
    
    if left_image is None or right_image is None:
        print(f"Error: Could not load images from {left_path} or {right_path}")
        return
    
    left_name = os.path.splitext(os.path.basename(left_path))[0]
    
    disp_map = compute_disparity(left_image, right_image, block_size, max_disparity)
    
    os.makedirs('maps', exist_ok=True)
    
    grayscale_filename = f'maps/disparity_{left_name}.png'
    cv2.imwrite(grayscale_filename, disp_map)
    print(f"Original disparity map saved as '{grayscale_filename}'")
    
    print("Generating colored disparity maps...")
    save_colormap_comparison(disp_map, f'maps/disparity_{left_name}')
    
    colored_jet = apply_colormap(disp_map, 'jet')
    colored_filename = f'maps/disparity_colored_{left_name}.png'
    cv2.imwrite(colored_filename, colored_jet)
    print(f"Recommended colored disparity map (JET) saved as '{colored_filename}'")

if __name__ == "__main__":
    process_stereo_pair('img/corridorl.jpg', 'img/corridorr.jpg', block_size=5, max_disparity=64)
    process_stereo_pair('img/triclopsi2l.jpg', 'img/triclopsi2r.jpg', block_size=5, max_disparity=64)