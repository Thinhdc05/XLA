"""
Utility functions - Các hàm tiện ích
"""

import cv2
import numpy as np
from scipy.signal import find_peaks


def auto_suggest_mode(gray):
    """Phân tích ảnh và gợi ý mode phù hợp"""
    brightness = gray.mean()
    contrast = gray.std()
    
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    gray_uint8 = gray.astype(np.uint8)
    glcm_range = gray_uint8.max() - gray_uint8.min()
    
    reasons = []
    scores = {
        'Standard': 0,
        'Hand-drawn': 0,
        'Photo': 0,
        'Low Quality': 0
    }
    
    if brightness < 70:
        scores['Low Quality'] += 3
        reasons.append("Ảnh tối")
    elif brightness > 200:
        scores['Standard'] += 2
        reasons.append("Ảnh sáng đều")
    else:
        scores['Photo'] += 1
    
    if contrast < 30:
        scores['Low Quality'] += 3
        reasons.append("Độ tương phản thấp")
    elif contrast > 70:
        scores['Hand-drawn'] += 2
        scores['Standard'] += 1
        reasons.append("Độ tương phản cao")
    else:
        scores['Photo'] += 1
    
    if edge_density > 0.20:
        scores['Hand-drawn'] += 4
        reasons.append("Nét đậm/vẽ tay")
    elif edge_density > 0.15:
        scores['Hand-drawn'] += 2
        scores['Photo'] += 1
    elif edge_density < 0.08:
        scores['Standard'] += 2
        reasons.append("Nét mảnh/scan")
    
    if laplacian_var < 100:
        scores['Low Quality'] += 3
        reasons.append("Có nhiễu/mờ")
    elif laplacian_var > 500:
        scores['Standard'] += 2
        scores['Hand-drawn'] += 1
    
    if glcm_range > 200:
        scores['Hand-drawn'] += 2
        scores['Photo'] += 1
        reasons.append("Độ sâu màu cao")
    
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten()
    peaks, _ = find_peaks(hist, height=hist.max() * 0.1, distance=50)
    if len(peaks) == 2:
        scores['Standard'] += 3
        reasons.append("Histogram 2 đỉnh (scan)")
    
    best_mode = max(scores, key=scores.get)
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if sorted_scores[0][1] - sorted_scores[1][1] < 2:
        if 'Photo' in [sorted_scores[0][0], sorted_scores[1][0]]:
            best_mode = 'Photo'
            reasons.append("(Chọn Photo vì điểm gần nhau)")
    
    reason_str = ", ".join(reasons[:2]) if reasons else "Phân tích tự động"
    
    return best_mode, reason_str, scores
