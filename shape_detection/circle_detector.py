# ==============================================================================
# CIRCLE DETECTION - Phát hiện hình tròn với OpenCV
# ==============================================================================

import cv2
import numpy as np

def detect_circles(image, method='hough'):
    """
    Phát hiện hình tròn bằng nhiều phương pháp
    
    Args:
        image: Grayscale or BGR image
        method: 'hough', 'contour', or 'hybrid'
    
    Returns:
        Dictionary với circles detected và properties
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    results = {
        'circles': [],
        'method': method,
        'visualization': None
    }
    
    if method == 'hough':
        # Hough Circle Transform
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=30,
            minRadius=5,
            maxRadius=100
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            results['circles'] = circles
            
            # Draw
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            for (x, y, r) in circles:
                cv2.circle(vis, (x, y), r, (0, 255, 0), 2)
                cv2.circle(vis, (x, y), 2, (0, 0, 255), 3)
            results['visualization'] = vis
    
    elif method == 'contour':
        # Contour-based with circularity
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        circles = []
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:  # Too small
                continue
            
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Check circularity (circle ≈ 1.0)
            if circularity > 0.75:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                circles.append({
                    'center': (int(x), int(y)),
                    'radius': int(radius),
                    'circularity': circularity,
                    'area': area
                })
                
                cv2.circle(vis, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                cv2.circle(vis, (int(x), int(y)), 2, (0, 0, 255), 3)
        
        results['circles'] = circles
        results['visualization'] = vis
    
    else:  # hybrid
        # Combine both methods
        hough_result = detect_circles(image, 'hough')
        contour_result = detect_circles(image, 'contour')
        
        # Merge results (prefer contour for better properties)
        results = contour_result
        results['method'] = 'hybrid'
    
    return results

def analyze_circle_properties(image):
    """
    Phân tích chi tiết properties của circle
    
    Args:
        image: Binary image chứa circle
    
    Returns:
        Dictionary với properties
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    cnt = max(contours, key=cv2.contourArea)
    
    # Fit circle
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    
    # Calculate circularity
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
    
    # Fit ellipse (if enough points)
    if len(cnt) >= 5:
        ellipse = cv2.fitEllipse(cnt)
        (ex, ey), (MA, ma), angle = ellipse
        eccentricity = np.sqrt(1 - (min(MA, ma) / max(MA, ma))**2) if max(MA, ma) > 0 else 0
    else:
        eccentricity = None
        ellipse = None
    
    return {
        'center': (int(x), int(y)),
        'radius': int(radius),
        'area': area,
        'perimeter': perimeter,
        'circularity': circularity,
        'eccentricity': eccentricity,
        'is_circle': circularity > 0.85,
        'ellipse': ellipse
    }

__all__ = ['detect_circles', 'analyze_circle_properties']
