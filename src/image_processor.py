"""
Image processing module - Xử lý ảnh cho Streamlit app
"""

import cv2
import numpy as np
from scipy.ndimage import center_of_mass
from preprocessing import MODES


def process_single_image(image_data, mode, params):
    """Xử lý 1 ảnh với mode và params - Lưu lại 8 bước"""
    img = np.array(image_data)
    
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    
    h, w = gray.shape
    
    step1_original = img
    step2_gray = gray
    step3_thresh = MODES[mode](gray, params)
    
    kernel_close = np.ones((3,3), np.uint8)
    step4_closed = cv2.morphologyEx(step3_thresh, cv2.MORPH_CLOSE, kernel_close)
    
    contours, _ = cv2.findContours(step4_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        largest = max(contours, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(largest)
        padding = 20
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w, x + cw + padding)
        y2 = min(h, y + ch + padding)
        step5_cropped = step4_closed[y1:y2, x1:x2]
    else:
        step5_cropped = step4_closed
    
    if params.get('dilation', 1) > 0:
        kernel_dilate = np.ones((2,2), np.uint8)
        step6_dilated = cv2.dilate(step5_cropped, kernel_dilate, iterations=params.get('dilation', 1))
    else:
        step6_dilated = step5_cropped
    
    if step6_dilated.size > 0 and step6_dilated.shape[0] > 0 and step6_dilated.shape[1] > 0:
        step7_resized = cv2.resize(step6_dilated, (20, 20), interpolation=cv2.INTER_AREA)
    else:
        step7_resized = np.zeros((20, 20), dtype=np.uint8)
    
    padded = np.pad(step7_resized, ((4,4),(4,4)), 'constant', constant_values=0)
    
    if np.sum(padded) > 0:
        cy, cx = center_of_mass(padded)
        shiftx = int(np.round(14 - cx))
        shifty = int(np.round(14 - cy))
        M = np.float32([[1, 0, shiftx], [0, 1, shifty]])
        step8_centered = cv2.warpAffine(padded, M, (28, 28))
    else:
        step8_centered = padded
    
    normalized = step8_centered.astype(np.float32) / 255.0
    processed = normalized.reshape(1, 28, 28, 1)
    
    return {
        'step1_original': step1_original,
        'step2_gray': step2_gray,
        'step3_thresh': step3_thresh,
        'step4_closed': step4_closed,
        'step5_cropped': step5_cropped,
        'step6_dilated': step6_dilated,
        'step7_resized': step7_resized,
        'step8_centered': step8_centered,
        'processed': processed,
        'original': step1_original,
        'gray': step2_gray,
        'thresh': step3_thresh,
        'cropped': step5_cropped,
        'final': step8_centered
    }


def predict_batch(model, files_data, modes, params_list, labels, progress_callback=None):
    """Dự đoán batch"""
    results = []
    n_total = len(files_data)
    
    for idx, (file_info, mode, params) in enumerate(zip(files_data, modes, params_list)):
        if progress_callback:
            progress_callback(idx + 1, n_total, f'Đang xử lý ảnh {idx+1}/{n_total}...')
        
        try:
            image = file_info['image']
            processed = process_single_image(image, mode, params)
            
            predictions = model.predict(processed['processed'], verbose=0)
            pred_class = np.argmax(predictions[0])
            confidence = predictions[0][pred_class] * 100
            
            results.append({
                'idx': idx + 1,
                'file_name': file_info['name'],
                'mode': mode,
                'predicted_class': pred_class,
                'predicted_label': labels[pred_class],
                'confidence': confidence,
                'predictions': predictions[0],
                'images': processed,
                'params': params,
                'error': False
            })
        except Exception as e:
            results.append({
                'idx': idx + 1,
                'file_name': file_info['name'],
                'mode': mode,
                'error': True,
                'error_message': str(e)
            })
    
    if progress_callback:
        progress_callback(n_total, n_total, '✓ Hoàn thành!')
    
    return results
