# ==============================================================================
# POLYGON DETECTION - Phát hiện Rectangle & Triangle
# ==============================================================================

import cv2
import numpy as np

def detect_rectangles(image):
    """
    Phát hiện hình chữ nhật/vuông
    
    Args:
        image: Grayscale or BGR image
    
    Returns:
        Dictionary với rectangles detected
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Threshold
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rectangles = []
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue
        
        # Approximate polygon
        perimeter = cv2.arcLength(cnt, True)
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        # Check if 4 vertices (rectangle/square)
        if len(approx) == 4:
            # Get bounding rect
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / h if h > 0 else 0
            
            # Check if close to rectangle (not too skewed)
            rect_area = w * h
            extent = area / rect_area if rect_area > 0 else 0
            
            if extent > 0.7:  # At least 70% filled
                is_square = 0.8 < aspect_ratio < 1.2
                
                rectangles.append({
                    'vertices': approx,
                    'bbox': (x, y, w, h),
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'extent': extent,
                    'is_square': is_square,
                    'type': 'Square' if is_square else 'Rectangle'
                })
                
                # Draw
                cv2.drawContours(vis, [approx], 0, (0, 255, 0), 2)
                cv2.putText(vis, 'Rect' if not is_square else 'Sq', 
                           (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return {
        'rectangles': rectangles,
        'count': len(rectangles),
        'visualization': vis
    }

def detect_triangles(image):
    """
    Phát hiện tam giác
    
    Args:
        image: Grayscale or BGR image
    
    Returns:
        Dictionary với triangles detected
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Threshold
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    triangles = []
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue
        
        # Approximate polygon
        perimeter = cv2.arcLength(cnt, True)
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        # Check if 3 vertices (triangle)
        if len(approx) == 3:
            # Calculate angles
            pts = approx.reshape(3, 2)
            angles = []
            
            for i in range(3):
                p1 = pts[i]
                p2 = pts[(i+1) % 3]
                p3 = pts[(i+2) % 3]
                
                v1 = p1 - p2
                v2 = p3 - p2
                
                angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))
                angles.append(np.degrees(angle))
            
            # Classify triangle type
            angles_sorted = sorted(angles)
            if max(angles_sorted) > 120:
                tri_type = 'Obtuse'  # Tù
            elif max(angles_sorted) > 91:
                tri_type = 'Right'  # Vuông
            elif all(85 < a < 95 for a in angles_sorted):
                tri_type = 'Equilateral'  # Đều
            else:
                tri_type = 'Acute'  # Nhọn
            
            triangles.append({
                'vertices': approx,
                'area': area,
                'angles': angles,
                'type': tri_type
            })
            
            # Draw
            cv2.drawContours(vis, [approx], 0, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(approx)
            cv2.putText(vis, 'Tri', (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return {
        'triangles': triangles,
        'count': len(triangles),
        'visualization': vis
    }

def hybrid_shape_detection(image):
    """
    Kết hợp detection cho tất cả shapes
    
    Args:
        image: Grayscale or BGR image
    
    Returns:
        Dictionary với all shapes detected
    """
    # Import circle detector
    from .circle_detector import detect_circles
    
    circles = detect_circles(image, method='contour')
    rectangles = detect_rectangles(image)
    triangles = detect_triangles(image)
    
    # Determine dominant shape
    n_circles = len(circles['circles']) if 'circles' in circles else 0
    n_rects = rectangles['count']
    n_tris = triangles['count']
    
    if n_circles > 0:
        dominant = 'Circle'
    elif n_rects > 0:
        dominant = 'Rectangle'
    elif n_tris > 0:
        dominant = 'Triangle'
    else:
        dominant = 'Unknown'
    
    # Combined visualization
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    # Draw all shapes with different colors
    if n_circles > 0:
        for circ in circles['circles']:
            if isinstance(circ, dict):
                x, y = circ['center']
                r = circ['radius']
            else:  # From hough
                x, y, r = circ
            cv2.circle(vis, (x, y), r, (0, 0, 255), 2)  # Red for circles
    
    if n_rects > 0:
        for rect in rectangles['rectangles']:
            cv2.drawContours(vis, [rect['vertices']], 0, (0, 255, 0), 2)  # Green for rectangles
    
    if n_tris > 0:
        for tri in triangles['triangles']:
            cv2.drawContours(vis, [tri['vertices']], 0, (255, 0, 0), 2)  # Blue for triangles
    
    return {
        'circles': circles,
        'rectangles': rectangles,
        'triangles': triangles,
        'dominant_shape': dominant,
        'visualization': vis,
        'summary': {
            'n_circles': n_circles,
            'n_rectangles': n_rects,
            'n_triangles': n_tris
        }
    }

__all__ = ['detect_rectangles', 'detect_triangles', 'hybrid_shape_detection']
