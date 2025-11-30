# ==============================================================================
# FEATURE MAPS VISUALIZATION - Xem CNN học gì từ ảnh
# Chứng minh hiểu trích chọn đặc trưng (YÊU CẦU ĐỀ BÀI)
# ==============================================================================

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import streamlit as st

# ==============================================================================
# 1. VISUALIZE INTERMEDIATE LAYERS
# ==============================================================================

def get_layer_outputs(model, image):
    """
    Lấy output của tất cả các layers
    
    Args:
        model: Keras model
        image: Input image (1, 28, 28, 1)
    
    Returns:
        Dictionary {layer_name: output}
    """
    # Create intermediate models
    layer_outputs = []
    layer_names = []
    
    for layer in model.layers:
        if 'conv' in layer.name.lower() or 'dense' in layer.name.lower() or 'pool' in layer.name.lower():
            layer_outputs.append(layer.output)
            layer_names.append(layer.name)
    
    if len(layer_outputs) == 0:
        return {}
    
    # Create model that outputs all intermediate layers
    intermediate_model = keras.Model(inputs=model.input, outputs=layer_outputs)
    
    # Get outputs
    outputs = intermediate_model.predict(image, verbose=0)
    
    # Create dictionary
    result = {}
    for name, output in zip(layer_names, outputs):
        result[name] = output
    
    return result

# ==============================================================================
# 2. VISUALIZE CONVOLUTIONAL FEATURE MAPS
# ==============================================================================

def visualize_feature_maps(model, image, layer_name=None):
    """
    Visualize feature maps của Conv layers
    
    Args:
        model: Keras model
        image: Input image (1, 28, 28, 1)
        layer_name: Tên layer cụ thể, None = all conv layers
    
    Returns:
        Dictionary với feature maps
    """
    layer_outputs = get_layer_outputs(model, image)
    
    feature_maps = {}
    
    for name, output in layer_outputs.items():
        if 'conv' in name.lower():
            if layer_name is None or name == layer_name:
                feature_maps[name] = output[0]  # Remove batch dimension
    
    return feature_maps

# ==============================================================================
# 3. VISUALIZE FILTERS (KERNELS)
# ==============================================================================

def visualize_filters(model, layer_name=None):
    """
    Visualize filters/kernels của Conv layers
    
    Args:
        model: Keras model
        layer_name: Tên layer cụ thể, None = first conv layer
    
    Returns:
        Dictionary với filters
    """
    filters_dict = {}
    
    for layer in model.layers:
        if 'conv' in layer.name.lower():
            if layer_name is None or layer.name == layer_name:
                weights = layer.get_weights()[0]  # (height, width, in_channels, out_channels)
                filters_dict[layer.name] = weights
                
                if layer_name is not None:
                    break
    
    return filters_dict

# ==============================================================================
# 4. ACTIVATION HEATMAP (Grad-CAM style)
# ==============================================================================

def generate_heatmap(model, image, pred_class):
    """
    Tạo heatmap cho vùng model "chú ý" nhất
    Inspired by Grad-CAM
    
    Args:
        model: Keras model
        image: Input image (1, 28, 28, 1)
        pred_class: Class index được predict
    
    Returns:
        Heatmap array
    """
    # Find last conv layer
    last_conv_layer = None
    for layer in reversed(model.layers):
        if 'conv' in layer.name.lower():
            last_conv_layer = layer
            break
    
    if last_conv_layer is None:
        return None
    
    # Create model from input to last conv layer
    grad_model = keras.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )
    
    # Compute gradient
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        loss = predictions[:, pred_class]
    
    # Gradient of the predicted class wrt the conv output
    grads = tape.gradient(loss, conv_outputs)
    
    # Average gradient over all feature maps
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weighted combination of feature maps
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    
    # Convert to numpy and normalize to [0, 1]
    heatmap = heatmap.numpy() if hasattr(heatmap, 'numpy') else heatmap
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-10)
    
    # Resize to match input
    heatmap = cv2.resize(heatmap, (28, 28))
    
    return heatmap

# ==============================================================================
# 5. PLOT FEATURE MAPS
# ==============================================================================

def plot_feature_maps(feature_maps, max_filters=16):
    """
    Plot feature maps dạng grid
    
    Args:
        feature_maps: Output từ visualize_feature_maps()
        max_filters: Số filter tối đa hiển thị
    
    Returns:
        Matplotlib figures dictionary
    """
    figures = {}
    
    for layer_name, fmap in feature_maps.items():
        n_features = min(fmap.shape[-1], max_filters)
        
        # Calculate grid size
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 3))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'Feature Maps: {layer_name} ({fmap.shape})', fontsize=14, fontweight='bold')
        
        for i in range(n_features):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].imshow(fmap[:, :, i], cmap='viridis')
            axes[row, col].set_title(f'Filter {i+1}', fontsize=10)
            axes[row, col].axis('off')
        
        # Hide empty subplots
        for i in range(n_features, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        figures[layer_name] = fig
    
    return figures

# ==============================================================================
# 6. PLOT FILTERS
# ==============================================================================

def plot_filters(filters, max_filters=32):
    """
    Plot filters/kernels dạng grid
    
    Args:
        filters: Output từ visualize_filters()
        max_filters: Số filter tối đa hiển thị
    
    Returns:
        Matplotlib figures dictionary
    """
    figures = {}
    
    for layer_name, weights in filters.items():
        # weights shape: (height, width, in_channels, out_channels)
        h, w, in_ch, out_ch = weights.shape
        
        n_filters = min(out_ch, max_filters)
        n_cols = 8
        n_rows = (n_filters + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 2))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'Filters: {layer_name} - Shape: {weights.shape}', fontsize=14, fontweight='bold')
        
        for i in range(n_filters):
            row = i // n_cols
            col = i % n_cols
            
            # Get filter for first input channel
            filter_img = weights[:, :, 0, i]
            
            # Normalize
            filter_img = (filter_img - filter_img.min()) / (filter_img.max() - filter_img.min() + 1e-10)
            
            axes[row, col].imshow(filter_img, cmap='gray')
            axes[row, col].set_title(f'F{i+1}', fontsize=8)
            axes[row, col].axis('off')
        
        # Hide empty subplots
        for i in range(n_filters, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        figures[layer_name] = fig
    
    return figures

# ==============================================================================
# 7. STREAMLIT DISPLAY HELPERS
# ==============================================================================

def display_feature_maps_streamlit(model, image, max_per_layer=8):
    """
    Hiển thị feature maps trong Streamlit
    
    Args:
        model: Keras model
        image: Input image (1, 28, 28, 1)
        max_per_layer: Số feature maps tối đa mỗi layer
    """
    st.markdown("### 🧠 FEATURE MAPS - CNN Học gì từ ảnh?")
    
    feature_maps = visualize_feature_maps(model, image)
    
    if len(feature_maps) == 0:
        st.warning("Model không có Conv layers")
        return
    
    for layer_name, fmap in feature_maps.items():
        with st.expander(f"📊 Layer: **{layer_name}** - Shape: {fmap.shape}", expanded=False):
            n_features = min(fmap.shape[-1], max_per_layer)
            
            st.caption(f"Hiển thị {n_features}/{fmap.shape[-1]} feature maps đầu tiên")
            
            # Display in grid
            cols = st.columns(4)
            for i in range(n_features):
                with cols[i % 4]:
                    st.image(fmap[:, :, i], width=150, clamp=True, channels='GRAY')
                    st.caption(f"Filter {i+1}")

def display_filters_streamlit(model, max_per_layer=16):
    """
    Hiển thị filters trong Streamlit
    
    Args:
        model: Keras model
        max_per_layer: Số filters tối đa mỗi layer
    """
    st.markdown("### 🔍 FILTERS (KERNELS) - Các bộ lọc CNN học được")
    
    filters = visualize_filters(model)
    
    if len(filters) == 0:
        st.warning("Model không có Conv layers")
        return
    
    for layer_name, weights in filters.items():
        h, w, in_ch, out_ch = weights.shape
        
        with st.expander(f"🎛️ Layer: **{layer_name}** - Shape: {weights.shape}", expanded=False):
            n_filters = min(out_ch, max_per_layer)
            
            st.caption(f"Hiển thị {n_filters}/{out_ch} filters. Mỗi filter {h}x{w} pixels.")
            
            # Display in grid
            cols = st.columns(8)
            for i in range(n_filters):
                with cols[i % 8]:
                    filter_img = weights[:, :, 0, i]
                    # Normalize
                    filter_img = (filter_img - filter_img.min()) / (filter_img.max() - filter_img.min() + 1e-10)
                    st.image(filter_img, width=60, clamp=True, channels='GRAY')
                    st.caption(f"{i+1}", help=f"Filter {i+1}")

def display_heatmap_streamlit(model, image, pred_class, original_image):
    """
    Hiển thị heatmap attention trong Streamlit
    
    Args:
        model: Keras model
        image: Input image (1, 28, 28, 1)
        pred_class: Predicted class index
        original_image: Original 28x28 image for overlay
    """
    st.markdown("### 🔥 ACTIVATION HEATMAP - Vùng model chú ý")
    
    heatmap = generate_heatmap(model, image, pred_class)
    
    if heatmap is None:
        st.warning("Không thể tạo heatmap (model không có Conv layer)")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image(original_image, width=200, channels='GRAY')
        st.caption("Ảnh gốc 28x28")
    
    with col2:
        st.image(heatmap, width=200, clamp=True, channels='GRAY')
        st.caption("Heatmap attention")
    
    with col3:
        # Overlay heatmap on original
        heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        original_rgb = cv2.cvtColor((original_image * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        overlay = cv2.addWeighted(original_rgb, 0.6, heatmap_colored, 0.4, 0)
        st.image(overlay, width=200, channels='RGB')
        st.caption("Overlay")
    
    st.info("🔥 Màu **đỏ/vàng** = vùng model chú ý nhiều nhất. Màu **xanh/tím** = ít quan trọng.")

# ==============================================================================
# EXPORT
# ==============================================================================

__all__ = [
    'get_layer_outputs',
    'visualize_feature_maps',
    'visualize_filters',
    'generate_heatmap',
    'plot_feature_maps',
    'plot_filters',
    'display_feature_maps_streamlit',
    'display_filters_streamlit',
    'display_heatmap_streamlit'
]
