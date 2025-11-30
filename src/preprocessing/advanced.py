# ==============================================================================
# ADVANCED PREPROCESSING - Kỹ thuật xử lý ảnh nâng cao
# Chứng minh hiểu sâu về xử lý ảnh
# ==============================================================================

import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage import center_of_mass
# from skimage.morphology import skeletonize, thin  # Optional - commented to avoid dependency
import matplotlib.pyplot as plt

# ==============================================================================
# 1. DESKEW - Xoay ảnh nghiêng về thẳng
# ==============================================================================

def deskew(image):
    """
    Deskew ảnh bằng cách tính góc nghiêng và xoay lại
    
    Args:
        image: Grayscale image (numpy array)
    
    Returns:
        Deskewed image, angle
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Invert if needed
    if np.mean(gray) > 127:
        gray = 255 - gray
    
    # Calculate moments
    moments = cv2.moments(gray)
    
    if abs(moments['mu02']) < 1e-2:
        return image, 0.0
    
    # Calculate skew angle
    skew = moments['mu11'] / moments['mu02']
    angle = 0.5 * np.arctan(2 * skew) * 180 / np.pi
    
    # Rotate
    h, w = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(gray, M, (w, h), 
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=0)
    
    return deskewed, angle

# ==============================================================================
# 2. THINNING - Làm mỏng nét viết
# ==============================================================================

def thin_image(image, method='morphology'):
    """
    Thinning - Làm mỏng nét viết (dùng OpenCV morphology)
    
    Args:
        image: Binary image
        method: 'morphology' (OpenCV erosion)
    
    Returns:
        Thinned image
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Ensure binary
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # OpenCV morphological thinning
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    thinned = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel, iterations=1)
    
    return thinned

# ==============================================================================
# 3. DISTANCE TRANSFORM - Phân tích độ dày
# ==============================================================================

def distance_transform_analysis(image):
    """
    Distance Transform - Tính khoảng cách đến biên gần nhất
    Hữu ích cho phân tích độ dày nét viết
    
    Args:
        image: Binary image
    
    Returns:
        dist_transform, max_dist, mean_dist
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Ensure binary
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Distance transform
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    
    # Normalize for visualization
    dist_normalized = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Statistics
    max_dist = np.max(dist_transform)
    mean_dist = np.mean(dist_transform[dist_transform > 0])
    
    return dist_normalized, max_dist, mean_dist

# ==============================================================================
# 4. CONTOUR SHAPE ANALYSIS - Phân tích hình dạng contour
# ==============================================================================

def contour_shape_analysis(image):
    """
    Phân tích chi tiết contour: area, perimeter, circularity, solidity, etc.
    
    Args:
        image: Binary or grayscale image
    
    Returns:
        Dictionary with shape properties
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Threshold
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    # Get largest contour
    contour = max(contours, key=cv2.contourArea)
    
    # Calculate properties
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    # Circularity: 4π * area / perimeter²
    # Perfect circle = 1.0
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
    
    # Convex hull
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    
    # Solidity: area / convex_hull_area
    solidity = area / hull_area if hull_area > 0 else 0
    
    # Bounding box
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h if h > 0 else 0
    extent = area / (w * h) if (w * h) > 0 else 0
    
    # Approximate polygon
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    num_vertices = len(approx)
    
    # Determine shape based on properties
    shape_type = "Unknown"
    if num_vertices == 3:
        shape_type = "Triangle"
    elif num_vertices == 4:
        shape_type = "Rectangle/Square"
    elif circularity > 0.85:
        shape_type = "Circle"
    elif num_vertices > 10:
        shape_type = "Circle/Ellipse"
    else:
        shape_type = f"Polygon ({num_vertices} sides)"
    
    return {
        'area': area,
        'perimeter': perimeter,
        'circularity': circularity,
        'solidity': solidity,
        'aspect_ratio': aspect_ratio,
        'extent': extent,
        'num_vertices': num_vertices,
        'shape_type': shape_type,
        'contour': contour,
        'approx_polygon': approx
    }

# ==============================================================================
# 5. HU MOMENTS - Đặc trưng bất biến
# ==============================================================================

def compute_hu_moments(image):
    """
    Tính Hu Moments - 7 invariant moments
    Bất biến với translation, rotation, scale
    
    Args:
        image: Binary or grayscale image
    
    Returns:
        7 Hu moments
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Threshold
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Calculate moments
    moments = cv2.moments(binary)
    
    # Calculate Hu moments
    hu_moments = cv2.HuMoments(moments).flatten()
    
    # Log scale (for better numerical stability)
    hu_moments_log = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
    
    return hu_moments_log

# ==============================================================================
# 6. EDGE DETECTION COMPARISON - So sánh các phương pháp edge
# ==============================================================================

def edge_detection_comparison(image):
    """
    So sánh nhiều phương pháp edge detection
    
    Args:
        image: Grayscale image
    
    Returns:
        Dictionary with different edge maps
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Denoise
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 1. Sobel
    sobel_x = cv2.Sobel(denoised, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(denoised, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 2. Canny
    canny = cv2.Canny(denoised, 50, 150)
    
    # 3. Laplacian
    laplacian = cv2.Laplacian(denoised, cv2.CV_64F)
    laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 4. Scharr
    scharr_x = cv2.Scharr(denoised, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(denoised, cv2.CV_64F, 0, 1)
    scharr = np.sqrt(scharr_x**2 + scharr_y**2)
    scharr = cv2.normalize(scharr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return {
        'sobel': sobel,
        'canny': canny,
        'laplacian': laplacian,
        'scharr': scharr
    }

# ==============================================================================
# 7. ADAPTIVE THRESHOLDING COMPARISON
# ==============================================================================

def threshold_comparison(image):
    """
    So sánh các phương pháp threshold
    
    Args:
        image: Grayscale image
    
    Returns:
        Dictionary with different threshold results
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Denoise
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    
    results = {}
    
    # 1. Global threshold (Otsu)
    _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    results['otsu'] = otsu
    
    # 2. Adaptive Mean
    adaptive_mean = cv2.adaptiveThreshold(denoised, 255, 
                                          cv2.ADAPTIVE_THRESH_MEAN_C,
                                          cv2.THRESH_BINARY_INV, 11, 2)
    results['adaptive_mean'] = adaptive_mean
    
    # 3. Adaptive Gaussian
    adaptive_gaussian = cv2.adaptiveThreshold(denoised, 255,
                                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY_INV, 11, 2)
    results['adaptive_gaussian'] = adaptive_gaussian
    
    # 4. Binary threshold (fixed)
    _, binary = cv2.threshold(denoised, 127, 255, cv2.THRESH_BINARY_INV)
    results['binary_fixed'] = binary
    
    return results

# ==============================================================================
# 8. MORPHOLOGICAL OPERATIONS SHOWCASE
# ==============================================================================

def morphology_showcase(image):
    """
    Showcase các phép toán morphology
    
    Args:
        image: Binary image
    
    Returns:
        Dictionary with morphology results
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Ensure binary
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    results = {}
    results['original'] = binary
    results['erosion'] = cv2.erode(binary, kernel, iterations=1)
    results['dilation'] = cv2.dilate(binary, kernel, iterations=1)
    results['opening'] = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    results['closing'] = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    results['gradient'] = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
    results['tophat'] = cv2.morphologyEx(binary, cv2.MORPH_TOPHAT, kernel)
    results['blackhat'] = cv2.morphologyEx(binary, cv2.MORPH_BLACKHAT, kernel)
    
    return results

# ==============================================================================
# EXPORT
# ==============================================================================

__all__ = [
    'deskew',
    'thin_image',
    'distance_transform_analysis',
    'contour_shape_analysis',
    'compute_hu_moments',
    'edge_detection_comparison',
    'threshold_comparison',
    'morphology_showcase'
]
