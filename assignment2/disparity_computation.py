import numpy as np
import cv2

# =============================================================
# Stereo Disparity – Baseline (fixed) + Improved Variants
# API contract: each function returns ONLY a single 8-bit disparity_map
# with values scaled to [0,255] for visualization, matching the
# original function's behavior.
# =============================================================

# ------------------------------
# Helpers
# ------------------------------

def _to_gray_int16(img: np.ndarray) -> np.ndarray:
    """Convert BGR/RGB/Gray to single-channel int16 for safe arithmetic."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.int16)


def _normalize_for_visualization(disp: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Return an 8-bit visualization of disparity. Invalid pixels are 0.
    disp is assumed to be in *pixels* (float32).
    """
    viz = np.zeros_like(disp, dtype=np.float32)
    if np.any(valid):
        vmax = float(np.max(disp[valid]))
        if vmax > 0:
            viz[valid] = disp[valid] / vmax * 255.0
    return viz.astype(np.uint8)


def _valid_mask(h: int, w: int, half_block: int, max_disparity: int) -> np.ndarray:
    valid = np.zeros((h, w), dtype=bool)
    valid[half_block:h - half_block, half_block:w - half_block - max_disparity] = True
    return valid


def _census(img_u8: np.ndarray, window: int = 5) -> np.ndarray:
    """Compute Census transform bitstring per pixel. Returns uint32 array.
    window must be odd and <= 7 for 32-bit safety (e.g., 5x5 -> 24 comparisons).
    """
    assert window % 2 == 1, "window must be odd"
    half = window // 2
    h, w = img_u8.shape
    out = np.zeros((h, w), dtype=np.uint32)
    center = img_u8
    bit = 0
    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            if dy == 0 and dx == 0:
                continue
            shifted = np.zeros_like(img_u8)
            src_y0 = max(0, -dy)
            src_y1 = min(h, h - dy)
            src_x0 = max(0, -dx)
            src_x1 = min(w, w - dx)
            dst_y0 = src_y0 + dy
            dst_y1 = src_y1 + dy
            dst_x0 = src_x0 + dx
            dst_x1 = src_x1 + dx
            shifted[dst_y0:dst_y1, dst_x0:dst_x1] = img_u8[src_y0:src_y1, src_x0:src_x1]
            out |= ((shifted >= center) << bit).astype(np.uint32)
            bit += 1
    return out


# =============================================================
# Unified Disparity Computation Function
# =============================================================

def compute_disparity(left_img, right_img, block_size: int = 5, max_disparity: int = 64, method: str = 'SSD'):
    """
    Compute disparity map between two stereo images using various block matching methods.
    
    Args:
        left_img: Left stereo image
        right_img: Right stereo image
        block_size: Size of the matching block (default: 5)
        max_disparity: Maximum disparity value to search (default: 64)
        method: Matching method to use (default: 'SSD')
            - 'SSD': Sum of Squared Differences
            - 'SAD': Sum of Absolute Differences
            - 'ZSAD': Zero-mean SAD
            - 'ZNCC': Zero-mean Normalized Cross-Correlation
            - 'CENSUS': Census Transform + Hamming distance
    
    Returns:
        8-bit disparity visualization map with values in [0, 255]
    """
    method = method.upper()
    
    # Validate method
    valid_methods = ['SSD', 'SAD', 'ZSAD', 'ZNCC', 'CENSUS']
    if method not in valid_methods:
        raise ValueError(f"Invalid method '{method}'. Must be one of {valid_methods}")
    
    # Preprocessing based on method
    if method == 'CENSUS':
        Lg = _to_gray_int16(left_img)
        Rg = _to_gray_int16(right_img)
        L = _census(Lg.astype(np.uint8), window=5)
        R = _census(Rg.astype(np.uint8), window=5)
        h, w = L.shape
    elif method == 'ZNCC':
        L = _to_gray_int16(left_img).astype(np.float32)
        R = _to_gray_int16(right_img).astype(np.float32)
        h, w = L.shape
    else:
        L = _to_gray_int16(left_img)
        R = _to_gray_int16(right_img)
        h, w = L.shape
    
    half = block_size // 2
    disp = np.zeros((h, w), dtype=np.float32)
    valid = _valid_mask(h, w, half, max_disparity)
    
    # Special constants
    eps = 1e-6 if method == 'ZNCC' else None
    
    for y in range(half, h - half):
        for x in range(half, w - half - max_disparity):
            y0, y1 = y - half, y + half + 1
            x0, x1 = x - half, x + half + 1
            
            best_cost = np.inf
            best_d = 0
            
            # Get left block
            Lb = L[y0:y1, x0:x1]
            
            # Preprocessing for specific methods
            if method == 'ZSAD':
                Lm = int(np.mean(Lb))
                Lbz = Lb - Lm
            elif method == 'ZNCC':
                Lm = Lb.mean(); Ls = Lb.std() + eps
                Ln = (Lb - Lm) / Ls
            elif method == 'CENSUS':
                left_patch = Lb
            
            for d in range(max_disparity + 1):
                xr0, xr1 = x0 - d, x1 - d
                if xr0 < 0:
                    break  # further d will be out-of-bounds
                
                Rb = R[y0:y1, xr0:xr1]
                
                # Compute cost based on method
                if method == 'SSD':
                    diff = Lb - Rb
                    cost = float(np.sum(diff * diff))
                elif method == 'SAD':
                    cost = float(np.sum(np.abs(Lb - Rb)))
                elif method == 'ZSAD':
                    Rbz = Rb - int(np.mean(Rb))
                    cost = float(np.sum(np.abs(Lbz - Rbz)))
                elif method == 'ZNCC':
                    Rm = Rb.mean(); Rs = Rb.std() + eps
                    Rn = (Rb - Rm) / Rs
                    corr = float(np.mean(Ln * Rn))  # [-1, 1]
                    cost = -corr
                elif method == 'CENSUS':
                    right_patch = Rb
                    xor = np.bitwise_xor(left_patch, right_patch)
                    bytes_view = xor.view(np.uint8).reshape(xor.shape + (4,))
                    cost = int(np.unpackbits(bytes_view, axis=-1).sum())  # Hamming distance
                
                if cost < best_cost:
                    best_cost = cost
                    best_d = d
            
            disp[y, x] = best_d
    
    # Normalize to 0–255 for visualization (invalid stays 0)
    return _normalize_for_visualization(disp, valid)


# =============================================================
# Enhanced Features: Sobel Preprocessing & Consistency Check
# =============================================================

def _sobel_filter(img: np.ndarray) -> np.ndarray:
    """
    Apply Sobel edge detection filter to enhance features for matching.
    Returns gradient magnitude image (optimized using OpenCV).
    """
    # Convert to uint8 if needed
    if img.dtype == np.int16:
        img = np.clip(img, 0, 255).astype(np.uint8)
    elif img.dtype == np.float32:
        img = np.clip(img, 0, 255).astype(np.uint8)
    elif img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Use OpenCV for fast Sobel computation
    sobel_x = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3)
    
    # Compute gradient magnitude
    gradient_magnitude = np.sqrt(sobel_x.astype(np.float32)**2 + sobel_y.astype(np.float32)**2)
    gradient_magnitude = np.clip(gradient_magnitude, 0, 255)
    
    return gradient_magnitude.astype(np.int16)


def _left_right_consistency_check(disp_left: np.ndarray, disp_right: np.ndarray, threshold: int = 1) -> np.ndarray:
    """
    Apply left-right consistency check to filter invalid disparity values.
    
    Args:
        disp_left: Left disparity map
        disp_right: Right disparity map  
        threshold: Maximum allowed difference (default: 1)
    
    Returns:
        Filtered left disparity map with invalid pixels set to 0
    """
    h, w = disp_left.shape
    disp_filtered = disp_left.copy()
    
    for y in range(h):
        for x in range(w):
            left_disp = int(disp_left[y, x])
            # Check if right matching position is valid
            if x - left_disp >= 0 and x - left_disp < w:
                right_disp = int(disp_right[y, x - left_disp])
                disp_diff = abs(left_disp - right_disp)
                if disp_diff > threshold:
                    disp_filtered[y, x] = 0
    
    return disp_filtered


def compute_disparity_enhanced(left_img, right_img, block_size: int = 5, max_disparity: int = 64, 
                                method: str = 'SSD', use_sobel: bool = True, 
                                use_consistency_check: bool = False):
    """
    Enhanced disparity computation with optional Sobel preprocessing and LR consistency check.
    
    Args:
        left_img: Left stereo image
        right_img: Right stereo image
        block_size: Size of the matching block (default: 5)
        max_disparity: Maximum disparity value to search (default: 64)
        method: Matching method ('SSD', 'SAD', 'ZSAD', 'ZNCC', 'CENSUS')
        use_sobel: Apply Sobel edge preprocessing (default: True)
        use_consistency_check: Apply left-right consistency check (default: True)
    
    Returns:
        8-bit disparity visualization map with values in [0, 255]
    """
    # Preprocessing: Apply Sobel if enabled
    if use_sobel:
        left_preprocessed = _sobel_filter(_to_gray_int16(left_img))
        right_preprocessed = _sobel_filter(_to_gray_int16(right_img))
        left_input = left_preprocessed.astype(np.uint8)
        right_input = right_preprocessed.astype(np.uint8)
    else:
        left_input = left_img
        right_input = right_img
    
    # Compute left disparity
    disp_left = compute_disparity(left_input, right_input, block_size, max_disparity, method)
    
    # Optional: Compute right disparity for consistency check
    if use_consistency_check:
        # Compute right disparity (swap left and right images)
        disp_right = compute_disparity(right_input, left_input, block_size, max_disparity, method)
        
        # Apply consistency check
        disp_left = _left_right_consistency_check(disp_left, disp_right, threshold=1)
    
    return disp_left


# =============================================================
# Legacy Compatibility Functions
# =============================================================

def compute_disparity_sad(left_img, right_img, block_size: int = 5, max_disparity: int = 64):
    """Block matching using Sum of Absolute Differences (SAD)."""
    return compute_disparity(left_img, right_img, block_size, max_disparity, method='SAD')


def compute_disparity_zsad(left_img, right_img, block_size: int = 7, max_disparity: int = 64):
    """Zero-mean SAD: subtract local means before absolute differences."""
    return compute_disparity(left_img, right_img, block_size, max_disparity, method='ZSAD')


def compute_disparity_zncc(left_img, right_img, block_size: int = 7, max_disparity: int = 64):
    """ZNCC (maximize correlation). We minimize negative correlation as cost."""
    return compute_disparity(left_img, right_img, block_size, max_disparity, method='ZNCC')


def compute_disparity_census(left_img, right_img, block_size: int = 5, max_disparity: int = 64):
    """Census transform + Hamming distance (robust to illumination changes)."""
    return compute_disparity(left_img, right_img, block_size, max_disparity, method='CENSUS')


def compute_disparity_improved(left_img, right_img, block_size: int = 11, max_disparity: int = 64):
    """Improved method using ZNCC with larger block size."""
    return compute_disparity(left_img, right_img, block_size, max_disparity, method='ZNCC')

