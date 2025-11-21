# ==============================================================================
# VISUALIZATION PIPELINE - Hiển thị từng bước xử lý ảnh
# Chứng minh hiểu quá trình xử lý ảnh (YÊU CẦU ĐỀ BÀI)
# ==============================================================================

import cv2
import numpy as np
from scipy.ndimage import center_of_mass
import matplotlib.pyplot as plt
import streamlit as st

# ==============================================================================
# PIPELINE VISUALIZATION - 10 BƯỚC CHI TIẾT
# ==============================================================================

def visualize_pipeline(image, mode='digit'):
    """
    Visualize toàn bộ pipeline xử lý ảnh với 10 bước chi tiết
    
    Args:
        image: Input image (PIL or numpy array)
        mode: 'digit' or 'shape' - Tối ưu khác nhau
    
    Returns:
        Dictionary chứa tất cả bước trung gian và thông tin
    """
    
    # Convert to numpy if needed
    if hasattr(image, 'mode'):  # PIL Image
        img = np.array(image)
    else:
        img = image.copy()
    
    pipeline = {}
    explanations = {}
    
    # ========== BƯỚC 1: ORIGINAL ==========
    pipeline['step1_original'] = img.copy()
    explanations['step1'] = {
        'title': 'Bước 1: Ảnh gốc (Original)',
        'description': 'Ảnh đầu vào từ camera/upload. Có thể màu hoặc grayscale, kích thước tùy ý.'
    }
    
    # ========== BƯỚC 2: GRAYSCALE ==========
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
    
    pipeline['step2_grayscale'] = gray
    explanations['step2'] = {
        'title': 'Bước 2: Chuyển grayscale',
        'description': f'Chuyển từ RGB (3 channels) sang grayscale (1 channel). Giảm dữ liệu, tăng tốc xử lý. Kích thước: {gray.shape}'
    }
    
    # ========== BƯỚC 3: CLAHE (Contrast Enhancement) ==========
    clahe = cv2.createCLAHE(clipLimit=2.0 if mode == 'digit' else 3.0, 
                            tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    pipeline['step3_clahe'] = enhanced
    explanations['step3'] = {
        'title': 'Bước 3: CLAHE - Tăng Contrast',
        'description': 'Contrast Limited Adaptive Histogram Equalization. Cải thiện contrast cục bộ, xử lý ánh sáng không đều. Đặc biệt tốt cho ảnh tối hoặc mờ.'
    }
    
    # ========== BƯỚC 4: DENOISING ==========
    if mode == 'shape':
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        denoise_method = 'Bilateral Filter (giữ edge)'
    else:
        denoised = cv2.GaussianBlur(enhanced, (5, 5), 0)
        denoise_method = 'Gaussian Blur'
    
    pipeline['step4_denoised'] = denoised
    explanations['step4'] = {
        'title': 'Bước 4: Khử nhiễu (Denoising)',
        'description': f'{denoise_method}. Loại bỏ nhiễu (noise) trước khi threshold. Làm mịn ảnh, giữ lại cấu trúc chính.'
    }
    
    # ========== BƯỚC 5: THRESHOLD ==========
    blocksize = 11 if mode == 'digit' else 13
    c_value = 2 if mode == 'digit' else 3
    
    thresh = cv2.adaptiveThreshold(
        denoised, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=blocksize, 
        C=c_value
    )
    
    pipeline['step5_threshold'] = thresh
    explanations['step5'] = {
        'title': 'Bước 5: Adaptive Threshold',
        'description': f'Chuyển sang ảnh nhị phân (binary). Adaptive threshold tự động tính ngưỡng cho từng vùng nhỏ. Tốt hơn threshold cố định. BlockSize={blocksize}, C={c_value}'
    }
    
    # ========== BƯỚC 6: MORPHOLOGY ==========
    kernel_size = 3 if mode == 'digit' else 5
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Closing: Lấp lỗ, kết nối nét
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Opening: Loại bỏ nhiễu nhỏ
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    
    pipeline['step6_morphology'] = cleaned
    explanations['step6'] = {
        'title': 'Bước 6: Morphology Operations',
        'description': f'Closing (lấp lỗ) + Opening (loại nhiễu). Kernel: {kernel_size}x{kernel_size} Ellipse. Làm sạch và kết nối contour.'
    }
    
    # ========== BƯỚC 7: CONTOUR DETECTION ==========
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours for visualization
    contour_img = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)
    if len(contours) > 0:
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(contour_img, [largest], 0, (255, 0, 0), 3)
    
    pipeline['step7_contours'] = contour_img
    explanations['step7'] = {
        'title': 'Bước 7: Phát hiện Contour',
        'description': f'Tìm biên (contour) của đối tượng. Tìm được {len(contours)} contours. Chọn contour lớn nhất (màu xanh dương) là đối tượng chính.'
    }
    
    # ========== BƯỚC 8: CROP & EXTRACT ==========
    h, w = cleaned.shape
    if len(contours) > 0:
        largest = max(contours, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(largest)
        
        padding = 15 if mode == 'digit' else 20
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w, x + cw + padding)
        y2 = min(h, y + ch + padding)
        
        cropped = cleaned[y1:y2, x1:x2]
    else:
        cropped = cleaned
    
    pipeline['step8_cropped'] = cropped
    explanations['step8'] = {
        'title': 'Bước 8: Crop & Extract',
        'description': f'Cắt (crop) đối tượng theo bounding box + padding. Loại bỏ background không cần thiết. Kích thước crop: {cropped.shape}'
    }
    
    # ========== BƯỚC 9: RESIZE TO 20x20 ==========
    if cropped.size > 0 and cropped.shape[0] > 0 and cropped.shape[1] > 0:
        resized = cv2.resize(cropped, (20, 20), interpolation=cv2.INTER_AREA)
    else:
        resized = np.zeros((20, 20), dtype=np.uint8)
    
    pipeline['step9_resized'] = resized
    explanations['step9'] = {
        'title': 'Bước 9: Resize về 20x20',
        'description': 'Resize về kích thước cố định 20x20 pixels. Sử dụng INTER_AREA (tốt cho downsampling). Chuẩn bị cho padding.'
    }
    
    # ========== BƯỚC 10: PAD TO 28x28 & CENTER ==========
    padded = np.pad(resized, ((4, 4), (4, 4)), 'constant', constant_values=0)
    
    # Center by center of mass
    if np.sum(padded) > 0:
        cy, cx = center_of_mass(padded)
        shiftx = int(np.round(14 - cx))
        shifty = int(np.round(14 - cy))
        M = np.float32([[1, 0, shiftx], [0, 1, shifty]])
        centered = cv2.warpAffine(padded, M, (28, 28))
    else:
        centered = padded
    
    pipeline['step10_final'] = centered
    explanations['step10'] = {
        'title': 'Bước 10: Pad & Center về 28x28',
        'description': f'Padding thêm 4 pixels mỗi bên: 20x20→28x28. Center bằng center of mass. Đây là input chuẩn cho CNN (28x28x1). Sẵn sàng cho prediction!'
    }
    
    # ========== NORMALIZED FOR MODEL ==========
    normalized = centered.astype(np.float32) / 255.0
    pipeline['normalized'] = normalized.reshape(1, 28, 28, 1)
    
    return {
        'pipeline': pipeline,
        'explanations': explanations,
        'final_input': pipeline['normalized']
    }

# ==============================================================================
# HISTOGRAM COMPARISON
# ==============================================================================

def show_histogram_comparison(original, processed):
    """
    So sánh histogram trước và sau xử lý
    
    Args:
        original: Original grayscale image
        processed: Processed grayscale image
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Original image
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Original histogram
    axes[0, 1].hist(original.ravel(), bins=256, range=[0, 256], color='blue', alpha=0.7)
    axes[0, 1].set_title('Original Histogram')
    axes[0, 1].set_xlabel('Pixel Intensity')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(alpha=0.3)
    
    # Processed image
    axes[1, 0].imshow(processed, cmap='gray')
    axes[1, 0].set_title('Processed Image')
    axes[1, 0].axis('off')
    
    # Processed histogram
    axes[1, 1].hist(processed.ravel(), bins=256, range=[0, 256], color='green', alpha=0.7)
    axes[1, 1].set_title('Processed Histogram')
    axes[1, 1].set_xlabel('Pixel Intensity')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    return fig

# ==============================================================================
# EDGE DETECTION VISUALIZATION
# ==============================================================================

def visualize_edge_detection(image):
    """
    Visualize nhiều phương pháp edge detection
    
    Args:
        image: Grayscale image
    
    Returns:
        Dictionary with edge images
    """
    # Sobel
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Canny
    canny = cv2.Canny(image, 50, 150)
    
    # Laplacian
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return {
        'sobel': sobel,
        'canny': canny,
        'laplacian': laplacian,
        'original': image
    }

# ==============================================================================
# STREAMLIT DISPLAY HELPERS
# ==============================================================================

def display_pipeline_streamlit(result):
    """
    Hiển thị pipeline trong Streamlit với layout đẹp
    
    Args:
        result: Output từ visualize_pipeline()
    """
    pipeline = result['pipeline']
    explanations = result['explanations']
    
    st.markdown("### 🔬 PIPELINE XỬ LÝ ẢNH - 10 BƯỚC CHI TIẾT")
    
    # Display in grid: 2 columns x 5 rows
    steps = [
        ('step1_original', 'step1'),
        ('step2_grayscale', 'step2'),
        ('step3_clahe', 'step3'),
        ('step4_denoised', 'step4'),
        ('step5_threshold', 'step5'),
        ('step6_morphology', 'step6'),
        ('step7_contours', 'step7'),
        ('step8_cropped', 'step8'),
        ('step9_resized', 'step9'),
        ('step10_final', 'step10'),
    ]
    
    for i in range(0, len(steps), 2):
        col1, col2 = st.columns(2)
        
        # Left column
        if i < len(steps):
            img_key, exp_key = steps[i]
            with col1:
                st.image(pipeline[img_key], width=250, channels='GRAY' if len(pipeline[img_key].shape) == 2 else 'RGB')
                st.markdown(f"**{explanations[exp_key]['title']}**")
                st.caption(explanations[exp_key]['description'])
        
        # Right column
        if i + 1 < len(steps):
            img_key, exp_key = steps[i + 1]
            with col2:
                st.image(pipeline[img_key], width=250, channels='GRAY' if len(pipeline[img_key].shape) == 2 else 'RGB')
                st.markdown(f"**{explanations[exp_key]['title']}**")
                st.caption(explanations[exp_key]['description'])

# ==============================================================================
# EXPORT
# ==============================================================================

__all__ = [
    'visualize_pipeline',
    'show_histogram_comparison',
    'visualize_edge_detection',
    'display_pipeline_streamlit'
]
