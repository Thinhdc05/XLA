# ==============================================================================
# APP HOÀN CHỈNH - Nhận dạng chữ số & hình học với XỬ LÝ ẢNH NÂNG CAO
# Đầy đủ tính năng cho BTL Xử lý Ảnh
# ==============================================================================

import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

# Import modules
from config import (
    DIGITS_LABELS, SHAPES_LABELS, 
    MODEL_PATH_DIGITS, MODEL_PATH_SHAPES
)
from scipy.ndimage import center_of_mass

# Import advanced modules
from preprocessing.visualizer import visualize_pipeline, display_pipeline_streamlit
from preprocessing.advanced import (contour_shape_analysis, compute_hu_moments,
                                   edge_detection_comparison, threshold_comparison)
from model_analysis.feature_maps import (display_feature_maps_streamlit, 
                                        display_filters_streamlit,
                                        display_heatmap_streamlit)
from shape_detection.polygon_detector import hybrid_shape_detection

# Canvas import
from streamlit_drawable_canvas import st_canvas

# ==============================================================================
# CONFIG
# ==============================================================================

st.set_page_config(
    page_title="BTL Xử Lý Ảnh - Nhận Dạng",
    page_icon="🔬",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
        max-width: 1400px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px;
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
    }
    .stButton button {
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# PREPROCESSING FUNCTION
# ==============================================================================

def smart_preprocess(image, mode='digit'):
    """Preprocessing thông minh cho cả digit và shape"""
    img = np.array(image)
    
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    
    h, w = gray.shape
    
    # 🆕 Tự động phát hiện nền sáng (Paint/Upload) vs nền tối (Canvas)
    mean_brightness = gray.mean()
    is_light_background = mean_brightness > 127
    
    if is_light_background:
        # Nền sáng (Paint): Đảo ngược để thành nền tối
        gray = 255 - gray
    
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0 if mode == 'digit' else 3.0, 
                            tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Denoise
    if mode == 'shape':
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
    else:
        denoised = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # Threshold
    blocksize = 11 if mode == 'digit' else 13
    c_value = 2 if mode == 'digit' else 3
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, blockSize=blocksize, C=c_value
    )
    
    # Morphology - Adaptive dựa trên độ dày nét
    kernel_size = 3 if mode == 'digit' else 5
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Phát hiện độ dày nét: nét mỏng = ít pixel trắng
    white_ratio = np.sum(thresh > 0) / thresh.size
    
    if white_ratio < 0.20:  # Nét mỏng (< 20% diện tích) - TĂNG từ 15% để bảo vệ tốt hơn
        # Morphology nhẹ: chỉ close, không open (giữ nét mỏng)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        cleaned = closed  # Bỏ qua OPEN để không mất nét
    else:  # Nét dày/bình thường - đa số trường hợp
        # Morphology đầy đủ: close + open (loại nhiễu tốt)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Lọc contours có diện tích đủ lớn (loại nhiễu nhỏ)
        min_area = (h * w) * 0.01  # Ít nhất 1% diện tích ảnh
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        if valid_contours:
            # Nếu có nhiều contours lớn (như số 8 bị tách), kiểm tra khoảng cách
            if len(valid_contours) > 1:
                # Tính bounding boxes của từng contour
                bboxes = [cv2.boundingRect(cnt) for cnt in valid_contours]
                
                # Kiểm tra xem các contours có gần nhau không (< 40% chiều rộng/cao ảnh)
                max_distance = max(h, w) * 0.4
                should_merge = False
                
                for i in range(len(bboxes)):
                    for j in range(i+1, len(bboxes)):
                        x1, y1, w1, h1 = bboxes[i]
                        x2, y2, w2, h2 = bboxes[j]
                        
                        # Khoảng cách giữa 2 centers
                        cx1, cy1 = x1 + w1//2, y1 + h1//2
                        cx2, cy2 = x2 + w2//2, y2 + h2//2
                        distance = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
                        
                        if distance < max_distance:
                            should_merge = True
                            break
                    if should_merge:
                        break
                
                if should_merge:
                    # Merge các contours gần nhau (số 8, 6, 9, 0)
                    all_points = np.vstack(valid_contours)
                    x, y, cw, ch = cv2.boundingRect(all_points)
                else:
                    # Các contours xa nhau → chọn lớn nhất (tránh nhiễu)
                    largest = max(valid_contours, key=cv2.contourArea)
                    x, y, cw, ch = cv2.boundingRect(largest)
            else:
                # Chỉ 1 contour - chọn nó
                x, y, cw, ch = cv2.boundingRect(valid_contours[0])
            
            padding = 15 if mode == 'digit' else 20
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(w, x + cw + padding)
            y2 = min(h, y + ch + padding)
            cropped = cleaned[y1:y2, x1:x2]
        else:
            cropped = cleaned
    else:
        cropped = cleaned
    
    # Resize
    if cropped.size > 0 and cropped.shape[0] > 0 and cropped.shape[1] > 0:
        resized = cv2.resize(cropped, (20, 20), interpolation=cv2.INTER_AREA)
    else:
        resized = np.zeros((20, 20), dtype=np.uint8)
    
    # Pad
    padded = np.pad(resized, ((4,4),(4,4)), 'constant', constant_values=0)
    
    # Center
    if np.sum(padded) > 0:
        cy, cx = center_of_mass(padded)
        shiftx = int(np.round(14 - cx))
        shifty = int(np.round(14 - cy))
        M = np.float32([[1, 0, shiftx], [0, 1, shifty]])
        centered = cv2.warpAffine(padded, M, (28, 28))
    else:
        centered = padded
    
    normalized = centered.astype(np.float32) / 255.0
    
    return {
        'original': img,
        'gray': gray,
        'enhanced': enhanced,
        'thresh': thresh,
        'cleaned': cleaned,
        'cropped': cropped,
        'final': centered,
        'processed': normalized.reshape(1, 28, 28, 1)
    }

# ==============================================================================
# LOAD MODELS
# ==============================================================================

@st.cache_resource
def load_digit_model():
    try:
        model = load_model(MODEL_PATH_DIGITS)
        return model, True
    except Exception as e:
        return None, False

@st.cache_resource
def load_shape_model():
    try:
        model = load_model(MODEL_PATH_SHAPES)
        return model, True
    except Exception as e:
        return None, False

digit_model, DIGIT_MODEL_LOADED = load_digit_model()
shape_model, SHAPE_MODEL_LOADED = load_shape_model()

# ==============================================================================
# SESSION STATE
# ==============================================================================

if 'recognition_mode' not in st.session_state:
    st.session_state.recognition_mode = 'Nhận dạng Số'
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'last_results' not in st.session_state:
    st.session_state.last_results = []
if 'show_results' not in st.session_state:
    st.session_state.show_results = False
if 'current_image_for_analysis' not in st.session_state:
    st.session_state.current_image_for_analysis = None

# ==============================================================================
# HEADER
# ==============================================================================

st.markdown("""
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 24px; border-radius: 12px; margin-bottom: 20px;'>
    <h1 style='color: white; margin: 0; font-size: 32px;'>
        🔬 BTL XỬ LÝ ẢNH - Nhận Dạng Chữ Số & Hình Học
    </h1>
    <p style='color: #f0f0f0; margin: 8px 0 0 0; font-size: 15px;'>
        ✨ Visualization Pipeline • Feature Maps • Shape Detection • Advanced CV Techniques
    </p>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# MAIN TABS
# ==============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Nhận Dạng", 
    "🔬 Pipeline Xử Lý Ảnh",
    "🧠 Feature Maps (CNN)",
    "🔍 Shape Analysis (OpenCV)",
    "📚 Kỹ Thuật Nâng Cao"
])

# ==============================================================================
# TAB 1: NHẬN DẠNG
# ==============================================================================

with tab1:
    st.markdown("### Nhận Dạng Tự Động")
    
    # Mode selection
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        if st.button("🔢 Nhận dạng SỐ (0-9)", 
                     type="primary" if st.session_state.recognition_mode == 'Nhận dạng Số' else "secondary",
                     use_container_width=True):
            st.session_state.recognition_mode = 'Nhận dạng Số'
            st.rerun()
    
    with col_m2:
        if st.button("🔺 Nhận dạng HÌNH HỌC", 
                     type="primary" if st.session_state.recognition_mode == 'Nhận dạng Hình học' else "secondary",
                     use_container_width=True):
            st.session_state.recognition_mode = 'Nhận dạng Hình học'
            st.rerun()
    
    # Display mode
    if st.session_state.recognition_mode == 'Nhận dạng Số':
        st.info("🔢 **Chế độ:** Nhận dạng số 0-9 | Model chuyên biệt cho digits")
        current_model = digit_model
        model_loaded = DIGIT_MODEL_LOADED
        current_labels = DIGITS_LABELS
        preprocess_mode = 'digit'
    else:
        st.info("🔺 **Chế độ:** Nhận dạng hình học (Tròn/HCN/Tam giác) | Model chuyên biệt cho shapes")
        current_model = shape_model
        model_loaded = SHAPE_MODEL_LOADED
        current_labels = SHAPES_LABELS
        preprocess_mode = 'shape'
    
    st.markdown("---")
    
    # Input method selection
    st.markdown("### Phương thức nhập ảnh")
    input_method = st.radio(
        "Chọn cách nhập:",
        ["Upload ảnh", "Vẽ trực tiếp"],
        horizontal=True
    )
    
    uploaded_file = None
    canvas_result = None
    
    if input_method == "Upload ảnh":
        uploaded_file = st.file_uploader("Chọn ảnh để nhận dạng", type=['png', 'jpg', 'jpeg'])
    else:
        st.markdown("**Vẽ chữ số hoặc hình học:**")
        
        # Chọn màu bút và độ dày
        col_color1, col_color2, col_color3 = st.columns([1, 1, 2])
        with col_color1:
            stroke_color = st.color_picker("Màu mực:", "#00FF00", key="stroke_color")
        with col_color2:
            stroke_width = st.slider("Độ dày nét:", 5, 30, 15, key="stroke_width")
        
        col_canvas1, col_canvas2 = st.columns([2, 1])
        
        with col_canvas1:
            # Drawing canvas
            canvas_result = st_canvas(
                fill_color="rgba(0, 0, 0, 0)",
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_color="#000000",
                height=280,
                width=280,
                drawing_mode="freedraw",
                key="canvas",
            )
        
        with col_canvas2:
            st.markdown("**💡 Mẹo vẽ tốt:**")
            if st.session_state.recognition_mode == 'Nhận dạng Số':
                st.write("• Vẽ chữ số 0-9")
                st.write("• Độ dày nét ≥ 8px")
                st.write("• Vẽ đủ lớn, rõ ràng")
                st.write("• Thẳng đứng (không xoay quá 20°)")
            else:
                st.write("• Tròn / Chữ nhật / Tam giác")
                st.write("• Độ dày nét ≥ 8px")
                st.write("• Có thể vẽ ở bất kỳ góc độ")
            
            if st.button("Xóa canvas", use_container_width=True):
                st.rerun()
    
    # Process input
    image = None
    
    if input_method == "Upload ảnh" and uploaded_file:
        image = Image.open(uploaded_file)
    elif input_method == "Vẽ trực tiếp" and canvas_result.image_data is not None:
        # Convert canvas to PIL Image (extract RGB, not alpha)
        canvas_data = canvas_result.image_data
        if np.sum(canvas_data) > 0:  # Check if drawn
            # Get RGB channels (colored strokes on black background)
            rgb_image = canvas_data[:, :, :3]
            
            # Store original RGB for display
            st.session_state['canvas_rgb'] = rgb_image.copy()
            
            # Convert to grayscale (this is the preprocessing step!)
            gray_canvas = cv2.cvtColor(rgb_image.astype('uint8'), cv2.COLOR_RGB2GRAY)
            
            # Store grayscale for display
            st.session_state['canvas_gray'] = gray_canvas.copy()
            
            # Create PIL Image
            image = Image.fromarray(gray_canvas)
    
    if image:
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            # Hiển thị ảnh RGB nếu từ canvas, grayscale nếu upload
            if input_method == "Vẽ trực tiếp" and 'canvas_rgb' in st.session_state:
                st.image(st.session_state['canvas_rgb'], caption="Ảnh gốc (RGB)", use_container_width=True)
            else:
                st.image(image, caption="Ảnh gốc", use_container_width=True, channels='GRAY')
        
        if st.button("NHẬN DẠNG", type="primary", use_container_width=True):
            if not model_loaded:
                st.error(f"✗ Model chưa train! Chạy: `python train_{'digits' if preprocess_mode=='digit' else 'shapes'}.py`")
            else:
                with st.spinner("Đang xử lý..."):
                    # Preprocess
                    processed = smart_preprocess(image, mode=preprocess_mode)
                    
                    # Predict
                    predictions = current_model.predict(processed['processed'], verbose=0)
                    pred_class = np.argmax(predictions[0])
                    confidence = predictions[0][pred_class] * 100
                    
                    # Save for other tabs (with RGB original if from canvas)
                    original_for_display = image
                    if input_method == "Vẽ trực tiếp" and 'canvas_rgb' in st.session_state:
                        # Use RGB canvas for pipeline display
                        original_for_display = Image.fromarray(st.session_state['canvas_rgb'])
                    
                    st.session_state.current_image_for_analysis = {
                        'original': original_for_display,
                        'processed': processed,
                        'prediction': pred_class,
                        'confidence': confidence,
                        'all_probs': predictions[0],
                        'mode': preprocess_mode
                    }
                    
                    with col2:
                        st.image(processed['final'], caption="Xử lý 28x28", use_container_width=True, channels='GRAY')
                    
                    with col3:
                        result_label = current_labels[pred_class]
                        color = '#4caf50' if confidence >= 90 else '#ff9800' if confidence >= 75 else '#f44336'
                        
                        st.markdown(f"""
                        <div style='background: {color}20; padding: 24px; border-radius: 12px; 
                                    border-left: 5px solid {color}; text-align: center;'>
                            <h2 style='color: {color}; margin: 0;'>{result_label}</h2>
                            <p style='color: {color}; font-size: 20px; margin: 10px 0 0 0;'>
                                Confidence: {confidence:.1f}%
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Top predictions
                    st.markdown("#### 📊 Top 3 Predictions")
                    top3_idx = np.argsort(predictions[0])[::-1][:3]
                    
                    cols = st.columns(3)
                    for i, idx in enumerate(top3_idx):
                        with cols[i]:
                            prob = predictions[0][idx] * 100
                            st.metric(
                                f"#{i+1}: {current_labels[idx]}", 
                                f"{prob:.1f}%",
                                delta=f"{'✓' if i == 0 else ''}"
                            )
                
                st.success("✅ Nhận dạng hoàn tất! Xem các tab khác để phân tích chi tiết.")

# ==============================================================================
# TAB 2: PIPELINE XỬ LÝ ẢNH
# ==============================================================================

with tab2:
    st.markdown("### Pipeline Xử Lý Ảnh (10 Bước)")
    
    if st.session_state.current_image_for_analysis:
        data = st.session_state.current_image_for_analysis
        
        # Visualize full pipeline
        result = visualize_pipeline(data['original'], mode=data['mode'])
        display_pipeline_streamlit(result)
        
        # Additional analysis
        st.markdown("---")
        st.markdown("### 📊 Phân Tích Histogram")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Histogram Ảnh Gốc**")
            gray_img = data['processed']['gray']
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            ax1.hist(gray_img.ravel(), bins=256, range=[0, 256], color='blue', alpha=0.7)
            ax1.set_xlabel('Pixel Intensity')
            ax1.set_ylabel('Frequency')
            ax1.grid(alpha=0.3)
            st.pyplot(fig1)
        
        with col2:
            st.markdown("**Histogram Sau CLAHE**")
            enhanced_img = data['processed']['enhanced']
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.hist(enhanced_img.ravel(), bins=256, range=[0, 256], color='green', alpha=0.7)
            ax2.set_xlabel('Pixel Intensity')
            ax2.set_ylabel('Frequency')
            ax2.grid(alpha=0.3)
            st.pyplot(fig2)
        
        st.caption("📈 CLAHE làm histogram phân bố đều hơn → Tăng contrast, dễ threshold")
        
    else:
        st.warning("⚠️ Vui lòng upload và nhận dạng ảnh ở Tab 1 trước!")

# ==============================================================================
# TAB 3: FEATURE MAPS
# ==============================================================================

with tab3:
    st.markdown("### Feature Maps - Trực Quan Hóa CNN")
    
    if st.session_state.current_image_for_analysis:
        data = st.session_state.current_image_for_analysis
        model = digit_model if data['mode'] == 'digit' else shape_model
        
        if model:
            # Feature Maps
            st.markdown("#### Activation Maps - Output của từng Conv Layer")
            display_feature_maps_streamlit(model, data['processed']['processed'], max_per_layer=8)
            
            st.markdown("---")
            
            # Filters
            st.markdown("#### Filters/Kernels - Bộ lọc CNN học được")
            display_filters_streamlit(model, max_per_layer=16)
            
            st.markdown("---")
            
            # Heatmap
            st.markdown("#### Attention Heatmap (Class Activation Map)")
            display_heatmap_streamlit(
                model, 
                data['processed']['processed'],
                data['prediction'],
                data['processed']['final']
            )
        else:
            st.error("Model chưa load!")
    else:
        st.warning("⚠️ Vui lòng upload và nhận dạng ảnh ở Tab 1 trước!")

# ==============================================================================
# TAB 4: SHAPE ANALYSIS
# ==============================================================================

with tab4:
    st.markdown("### Shape Detection - Phát Hiện Hình Học")
    
    if st.session_state.current_image_for_analysis:
        data = st.session_state.current_image_for_analysis
        
        # Hybrid detection
        st.markdown("#### Phát Hiện Tự Động (Hybrid Method)")
        
        gray_img = data['processed']['gray']
        result = hybrid_shape_detection(gray_img)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(gray_img, caption="Ảnh gốc (grayscale)", use_container_width=True, channels='GRAY')
        
        with col2:
            if result['visualization'] is not None:
                st.image(result['visualization'], caption="Phát hiện shape", use_container_width=True, channels='RGB')
        
        # Summary
        st.markdown("#### 📊 Kết Quả Phát Hiện")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Dominant Shape", result['dominant_shape'])
        with col2:
            st.metric("Circles", result['summary']['n_circles'])
        with col3:
            st.metric("Rectangles", result['summary']['n_rectangles'])
        with col4:
            st.metric("Triangles", result['summary']['n_triangles'])
        
        # Detailed analysis
        st.markdown("---")
        st.markdown("#### 🔬 Phân Tích Chi Tiết Contour")
        
        analysis = contour_shape_analysis(gray_img)
        
        if analysis:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Area", f"{analysis['area']:.0f} px²")
                st.metric("Perimeter", f"{analysis['perimeter']:.0f} px")
            
            with col2:
                st.metric("Circularity", f"{analysis['circularity']:.3f}")
                st.metric("Solidity", f"{analysis['solidity']:.3f}")
            
            with col3:
                st.metric("Aspect Ratio", f"{analysis['aspect_ratio']:.2f}")
                st.metric("Vertices", analysis['num_vertices'])
            
            st.success(f"**Kết luận:** {analysis['shape_type']}")
            
            # Hu Moments
            st.markdown("#### Hu Moments (Đặc trưng bất biến)")
            hu = compute_hu_moments(gray_img)
            
            hu_df = pd.DataFrame({
                'Moment': [f'Hu[{i}]' for i in range(7)],
                'Value': [f"{val:.4f}" for val in hu]
            })
            
            st.dataframe(hu_df, use_container_width=True, hide_index=True)
            st.caption("📌 Hu Moments bất biến với translation, rotation, scale - Dùng để nhận dạng shape")
        
    else:
        st.warning("⚠️ Vui lòng upload và nhận dạng ảnh ở Tab 1 trước!")

# ==============================================================================
# TAB 5: KỸ THUẬT NÂNG CAO
# ==============================================================================

with tab5:
    st.markdown("### So Sánh & Phân Tích Kỹ Thuật")
    
    # Phần 1: Kỹ thuật ĐÃ ÁP DỤNG trong hệ thống
    st.markdown("#### Tổng hợp kỹ thuật đã áp dụng")
    
    applied_techniques = pd.DataFrame({
        'STT': range(1, 11),
        'Kỹ Thuật': [
            'CLAHE (Contrast Limited AHE)',
            'Adaptive Threshold (Gaussian)',
            'Bilateral Filter / Gaussian Blur',
            'Morphology Operations (CLOSE/OPEN)',
            'Contour Detection (RETR_EXTERNAL)',
            'Bounding Box & Crop',
            'Center of Mass Alignment',
            'Distance Transform',
            'Hu Moments',
            'CNN Feature Extraction'
        ],
        'Áp dụng tại': [
            'Bước 2 - Pipeline',
            'Bước 4 - Pipeline',
            'Bước 3 - Pipeline',
            'Bước 5 - Pipeline',
            'Bước 6 - Pipeline',
            'Bước 7 - Pipeline',
            'Bước 9 - Pipeline',
            'Tab 4 - Shape Detection',
            'Tab 4 - Shape Detection',
            'Tab 3 - Feature Maps'
        ]
    })
    
    st.dataframe(applied_techniques, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Phần 2: SO SÁNH CHI TIẾT VÀ PHÂN TÍCH
    st.markdown("#### So sánh chi tiết các phương pháp")
    
    if st.session_state.current_image_for_analysis:
        data = st.session_state.current_image_for_analysis
        gray_img = data['processed']['gray']
        binary_img = data['processed']['cleaned']  # Ảnh sau morphology
        
        # =================================================================
        # 1. THRESHOLD COMPARISON (QUAN TRỌNG NHẤT)
        # =================================================================
        st.markdown("**1. Phương pháp Threshold - Tại sao chọn Adaptive Gaussian?**")
        
        thresholds = threshold_comparison(gray_img)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.image(thresholds['otsu'], caption="Otsu (Global)", use_container_width=True, channels='GRAY')
        with col2:
            st.image(thresholds['adaptive_mean'], caption="Adaptive Mean", use_container_width=True, channels='GRAY')
        with col3:
            st.image(thresholds['adaptive_gaussian'], caption="Adaptive Gaussian", use_container_width=True, channels='GRAY')
        with col4:
            st.image(thresholds['binary_fixed'], caption="Binary (threshold=127)", use_container_width=True, channels='GRAY')
        
        # Phân tích chi tiết
        st.markdown("**Phân tích so sánh:**")
        
        comparison_df = pd.DataFrame({
            'Phương pháp': ['Otsu (Global)', 'Binary Fixed', 'Adaptive Mean', 'Adaptive Gaussian (✓ Đã chọn)'],
            'Ưu điểm': [
                'Tự động tìm ngưỡng tối ưu toàn cục',
                'Đơn giản, nhanh nhất',
                'Thích nghi với từng vùng cục bộ',
                'Thích nghi cục bộ + Smoothing Gaussian'
            ],
            'Nhược điểm': [
                'Thất bại với ảnh ánh sáng không đều',
                'Không linh hoạt, phụ thuộc ngưỡng cố định',
                'Nhạy nhiễu, dễ tạo "salt-pepper"',
                'Chậm hơn Global, cần tune blockSize'
            ],
            'Phù hợp': [
                'Ảnh đồng nhất, ánh sáng đều',
                'Ảnh binary rõ ràng',
                'Ảnh có vùng tối/sáng khác nhau',
                'Ảnh viết tay, ánh sáng không đều (✓)'
            ]
        })
        
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        st.success("**Kết luận:** Chọn **Adaptive Gaussian** vì ảnh input (viết tay/vẽ) thường có ánh sáng không đều, bóng đổ. Gaussian smoothing giảm nhiễu tốt hơn Mean.")
        
        st.markdown("---")
        
        # =================================================================
        # 2. EDGE DETECTION vs CONTOUR DETECTION
        # =================================================================
        st.markdown("**2. Edge Detection vs Contour Detection - Tại sao chọn Contour?**")
        
        edges = edge_detection_comparison(gray_img)
        
        # Thêm cột Contour Detection (đang dùng)
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_img = np.zeros_like(binary_img)
        cv2.drawContours(contour_img, contours, -1, 255, 2)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.image(edges['sobel'], caption="Sobel", use_container_width=True, channels='GRAY')
        with col2:
            st.image(edges['canny'], caption="Canny", use_container_width=True, channels='GRAY')
        with col3:
            st.image(edges['laplacian'], caption="Laplacian", use_container_width=True, channels='GRAY')
        with col4:
            st.image(edges['scharr'], caption="Scharr", use_container_width=True, channels='GRAY')
        with col5:
            st.image(contour_img, caption="Contour (✓ Đang dùng)", use_container_width=True, channels='GRAY')
        
        # Phân tích chi tiết
        st.markdown("**Phân tích so sánh:**")
        
        edge_comparison_df = pd.DataFrame({
            'Phương pháp': ['Sobel', 'Canny', 'Laplacian', 'Scharr', 'Contour (✓ Đã chọn)'],
            'Loại': ['Gradient', 'Multi-stage', 'Second derivative', 'Gradient', 'Topology-based'],
            'Output': ['Edge pixels', 'Edge pixels', 'Edge pixels', 'Edge pixels', 'Closed curves (polygons)'],
            'Ưu điểm': [
                'Đơn giản, phát hiện nhanh',
                'Edge mỏng, rõ ràng nhất',
                'Nhạy với noise, phát hiện góc',
                'Chính xác hơn Sobel (kernel lớn)',
                'Cho bounding box, area, perimeter'
            ],
            'Nhược điểm': [
                'Edge dày, nhiễu',
                'Phức tạp, cần 2 threshold',
                'Rất nhạy nhiễu, edge kép',
                'Tính toán chậm hơn',
                'Cần ảnh binary sạch trước'
            ],
            'Phù hợp': [
                'Phát hiện biên nhanh',
                'Ảnh chất lượng cao, ít nhiễu',
                'Phát hiện góc, điểm đặc biệt',
                'Cần độ chính xác cao',
                'Phân tích hình học, tính toán đặc trưng (✓)'
            ]
        })
        
        st.dataframe(edge_comparison_df, use_container_width=True, hide_index=True)
        
        st.success("**Kết luận:** Chọn **Contour Detection** vì cần trích xuất **bounding box, area, perimeter** để crop & resize đối tượng. Edge detection chỉ cho pixels rời rạc, không thể tính toán đặc trưng hình học.")
        
        st.markdown("---")
        
        # =================================================================
        # 3. MORPHOLOGY PARAMETERS
        # =================================================================
        st.markdown("**3. Morphology Operations - Tại sao iterations=1, kernel=3x3?**")
        
        # Test với các config khác nhau
        kernel_3 = np.ones((3, 3), np.uint8)
        kernel_5 = np.ones((5, 5), np.uint8)
        
        morph_1_3 = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel_3, iterations=1)
        morph_2_3 = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel_3, iterations=2)
        morph_1_5 = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel_5, iterations=1)
        morph_2_5 = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel_5, iterations=2)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.image(morph_1_3, caption="iter=1, kernel=3x3 (✓)", use_container_width=True, channels='GRAY')
        with col2:
            st.image(morph_2_3, caption="iter=2, kernel=3x3", use_container_width=True, channels='GRAY')
        with col3:
            st.image(morph_1_5, caption="iter=1, kernel=5x5", use_container_width=True, channels='GRAY')
        with col4:
            st.image(morph_2_5, caption="iter=2, kernel=5x5", use_container_width=True, channels='GRAY')
        
        # Phân tích
        st.markdown("**Phân tích so sánh:**")
        
        morph_comparison_df = pd.DataFrame({
            'Cấu hình': ['iter=1, k=3x3 (✓)', 'iter=2, k=3x3', 'iter=1, k=5x5', 'iter=2, k=5x5'],
            'Kết nối nét': ['Tốt', 'Rất tốt', 'Tốt', 'Rất tốt'],
            'Bảo toàn góc': ['Xuất sắc (✓)', 'Trung bình', 'Kém', 'Rất kém'],
            'Độ dày nét': ['Vừa phải', 'Dày', 'Dày', 'Rất dày'],
            'Phù hợp': ['Hình học (góc vuông, nhọn)', 'Chữ số (curves)', 'Chữ cái lớn', 'Đối tượng to, nhiễu mạnh']
        })
        
        st.dataframe(morph_comparison_df, use_container_width=True, hide_index=True)
        
        st.success("**Kết luận:** Chọn **iterations=1, kernel=3x3** cho hình học để bảo toàn góc vuông (hình chữ nhật) và góc nhọn (tam giác). Iterations/kernel cao làm tròn góc → nhầm lẫn giữa các hình.")
        
    else:
        st.warning("Vui lòng upload và nhận dạng ảnh ở Tab 1 để xem so sánh chi tiết")

# ==============================================================================
# FOOTER
# ==============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <h4>BTL Xử Lý Ảnh - Nhận Dạng với CNN & OpenCV</h4>
    <p style='margin: 10px 0; font-size: 14px;'>
        Ứng dụng 10+ kỹ thuật xử lý ảnh: CLAHE, Adaptive Threshold, Morphology Operations,<br>
        Contour Detection, Distance Transform, Hu Moments, Center of Mass, CNN Feature Extraction
    </p>
</div>
""", unsafe_allow_html=True)
