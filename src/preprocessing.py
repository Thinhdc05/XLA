"""
Preprocessing module - Tiền xử lý ảnh
Chứa tất cả các hàm xử lý ảnh cho cả training và app
"""

import cv2
import numpy as np
from scipy.ndimage import center_of_mass


def preprocess_to_mnist(img_path):
    """Tiền xử lý ảnh về định dạng MNIST 28x28"""
    img = cv2.imread(img_path)
    if img is None: 
        return None

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, blockSize=9, C=2
    )
    
    kernel_close = np.ones((3,3), np.uint8)
    closed = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel_close)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center_h = int(h * 0.95)
    center_w = int(w * 0.95)
    start_y = (h - center_h) // 2
    start_x = (w - center_w) // 2
    center_rect = (start_x, start_y, center_w, center_h) 

    def is_fully_in_center(cnt, rect, min_area=100):
        x_c, y_c, cw_c, ch_c = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        area_ok = area > min_area
        x_ok = x_c >= rect[0] and (x_c + cw_c) <= (rect[0] + rect[2])
        y_ok = y_c >= rect[1] and (y_c + ch_c) <= (rect[1] + rect[3])
        return area_ok and x_ok and y_ok

    central_contours = [cnt for cnt in contours if is_fully_in_center(cnt, center_rect)]
    cropped = adaptive_thresh.copy()
    
    if central_contours:
        x_union_min, y_union_min = w, h
        x_union_max, y_union_max = 0, 0
        for cnt in central_contours:
            x_c, y_c, cw_c, ch_c = cv2.boundingRect(cnt)
            x_union_min = min(x_union_min, x_c)
            y_union_min = min(y_union_min, y_c)
            x_union_max = max(x_union_max, x_c + cw_c)
            y_union_max = max(y_union_max, y_c + ch_c)

        padding = 10 
        final_x1 = max(0, x_union_min - padding)
        final_y1 = max(0, y_union_min - padding)
        final_x2 = min(w, x_union_max + padding)
        final_y2 = min(h, y_union_max + padding)

        if final_x2 > final_x1 and final_y2 > final_y1:
            cropped = adaptive_thresh[final_y1:final_y2, final_x1:final_x2]

    kernel = np.ones((2,2), np.uint8)
    thickened = cv2.dilate(cropped, kernel, iterations=1)
    
    if thickened.size == 0 or thickened.shape[0] == 0 or thickened.shape[1] == 0:
        resized = np.zeros((20, 20), dtype=np.uint8)
    else:
        resized = cv2.resize(thickened, (20, 20), interpolation=cv2.INTER_AREA)

    mnist_like = np.pad(resized, ((4,4),(4,4)), 'constant', constant_values=0)

    if np.sum(mnist_like) > 0:
        cy, cx = center_of_mass(mnist_like)
        shiftx = int(np.round(mnist_like.shape[1]/2.0 - cx))
        shifty = int(np.round(mnist_like.shape[0]/2.0 - cy))
        M = np.float32([[1, 0, shiftx], [0, 1, shifty]])
        mnist_like_centered = cv2.warpAffine(mnist_like, M, (28, 28))
    else:
        mnist_like_centered = np.zeros((28, 28), dtype=np.uint8)
    
    return mnist_like_centered.astype(np.float32) / 255.0


# ==============================================================================
# PREPROCESSING MODES (cho app.py)
# ==============================================================================

def preprocess_standard(gray, params):
    """Standard - Ảnh scan/chụp rõ"""
    blurred = cv2.GaussianBlur(gray, (params['blur'], params['blur']), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        blockSize=params['block_size'], 
        C=params['c_value']
    )
    if params.get('opening', False):
        kernel = np.ones((2,2), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return thresh


def preprocess_handdrawn(gray, params):
    """Hand-drawn - Vẽ tay, nét đậm"""
    if params.get('denoise', False):
        gray = cv2.fastNlMeansDenoising(gray, None, params.get('denoise_h', 10), 7, 21)
    
    blurred = cv2.GaussianBlur(gray, (params.get('blur', 7), params.get('blur', 7)), 0)
    
    if params.get('use_otsu', True):
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=params.get('block_size', 11),
            C=params.get('c_value', 2)
        )
    
    kernel = np.ones((params.get('morph_size', 3), params.get('morph_size', 3)), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=params.get('close_iter', 2))
    
    return thresh


def preprocess_photo(gray, params):
    """Photo - Chụp từ camera, có thể nghiêng/ánh sáng không đều"""
    blurred = cv2.GaussianBlur(gray, (params.get('blur', 7), params.get('blur', 7)), 0)
    
    if params.get('use_clahe', True):
        clahe = cv2.createCLAHE(
            clipLimit=params.get('clahe_clip', 2.0), 
            tileGridSize=(params.get('clahe_grid', 8), params.get('clahe_grid', 8))
        )
        enhanced = clahe.apply(blurred)
    else:
        enhanced = blurred
    
    thresh = cv2.adaptiveThreshold(
        enhanced, 255, 
        cv2.ADAPTIVE_THRESH_MEAN_C if params.get('use_mean', True) else cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 
        blockSize=params.get('block_size', 15), 
        C=params.get('c_value', 5)
    )
    
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=params.get('close_iter', 1))
    
    return thresh


def preprocess_lowquality(gray, params):
    """Low Quality - Ảnh mờ/tối/nhiễu"""
    denoised = cv2.fastNlMeansDenoising(
        gray, None, 
        params.get('denoise_h', 10), 
        params.get('denoise_template', 7), 
        params.get('denoise_search', 21)
    )
    
    clahe = cv2.createCLAHE(
        clipLimit=params.get('clahe_clip', 3.0), 
        tileGridSize=(params.get('clahe_grid', 8), params.get('clahe_grid', 8))
    )
    enhanced = clahe.apply(denoised)
    
    blurred = cv2.GaussianBlur(enhanced, (params.get('blur', 5), params.get('blur', 5)), 0)
    
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        blockSize=params.get('block_size', 11), 
        C=params.get('c_value', 3)
    )
    
    kernel = np.ones((params.get('morph_size', 3), params.get('morph_size', 3)), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=params.get('close_iter', 2))
    
    return thresh


MODES = {
    'Standard': preprocess_standard,
    'Hand-drawn': preprocess_handdrawn,
    'Photo': preprocess_photo,
    'Low Quality': preprocess_lowquality
}
