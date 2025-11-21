"""
Tạo thêm synthetic shape data với đa dạng góc xoay và biến thể
Đặc biệt tập trung vào hình chữ nhật xoay (hình thoi)
"""
import numpy as np
import cv2
from scipy.ndimage import rotate
import matplotlib.pyplot as plt

def create_circle(size=200):
    """Tạo hình tròn"""
    img = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    radius = size // 3
    cv2.circle(img, (center, center), radius, 255, -1)
    return img

def create_rectangle(size=200, aspect_ratio=1.5):
    """Tạo hình chữ nhật với tỉ lệ khác nhau"""
    img = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    width = int(size // 3 * aspect_ratio)
    height = int(size // 3)
    
    x1 = center - width // 2
    y1 = center - height // 2
    x2 = center + width // 2
    y2 = center + height // 2
    
    cv2.rectangle(img, (x1, y1), (x2, y2), 255, -1)
    return img

def create_triangle(size=200, type='equilateral'):
    """Tạo tam giác với các loại khác nhau"""
    img = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    
    if type == 'equilateral':
        # Tam giác đều
        pts = np.array([
            [center, center - size//3],
            [center - size//3, center + size//4],
            [center + size//3, center + size//4]
        ], np.int32)
    elif type == 'right':
        # Tam giác vuông
        pts = np.array([
            [center - size//3, center - size//3],
            [center - size//3, center + size//3],
            [center + size//3, center + size//3]
        ], np.int32)
    else:  # isosceles
        # Tam giác cân
        pts = np.array([
            [center, center - size//3],
            [center - size//4, center + size//3],
            [center + size//4, center + size//3]
        ], np.int32)
    
    cv2.fillPoly(img, [pts], 255)
    return img

def add_noise(img, noise_level=10):
    """Thêm noise để đa dạng hóa"""
    noise = np.random.randn(*img.shape) * noise_level
    noisy = np.clip(img + noise, 0, 255).astype(np.uint8)
    return noisy

def generate_augmented_shapes(n_per_class=500):
    """
    Tạo synthetic shapes với augmentation đa dạng
    Đặc biệt tập trung vào hình chữ nhật xoay nhiều góc
    """
    X_synthetic = []
    y_synthetic = []
    
    print(f"🔧 Generating {n_per_class} samples per class...")
    
    # Class 0: Circle (Tròn)
    print("  Generating Circles...")
    for i in range(n_per_class):
        img = create_circle(size=200)
        
        # Random transformations
        angle = np.random.uniform(-180, 180)
        img = rotate(img, angle, reshape=False, cval=0)
        
        # Scale variations
        scale = np.random.uniform(0.7, 1.3)
        h, w = img.shape
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h))
        
        # Pad/crop to original size
        if new_h > h:
            img = img[:h, :w]
        else:
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            img = cv2.copyMakeBorder(img, pad_h, h-new_h-pad_h, pad_w, w-new_w-pad_w, 
                                    cv2.BORDER_CONSTANT, value=0)
        
        # Add noise
        if np.random.random() < 0.3:
            img = add_noise(img, noise_level=15)
        
        X_synthetic.append(img)
        y_synthetic.append(0)
    
    # Class 1: Rectangle (Hình chữ nhật) - NHIỀU GÓC XOAY
    print("  Generating Rectangles (including rotated/diamond shapes)...")
    for i in range(n_per_class):
        # Varied aspect ratios
        aspect_ratio = np.random.uniform(1.2, 2.0)
        img = create_rectangle(size=200, aspect_ratio=aspect_ratio)
        
        # ĐẶC BIỆT: Nhiều góc xoay, tập trung vào 30-60° (hình thoi)
        if i < n_per_class // 3:
            # 1/3 samples: xoay 30-60° (hình thoi)
            angle = np.random.uniform(30, 60)
        elif i < 2 * n_per_class // 3:
            # 1/3 samples: xoay nhẹ 0-30°
            angle = np.random.uniform(-30, 30)
        else:
            # 1/3 samples: xoay nhiều 60-90°
            angle = np.random.uniform(60, 90)
        
        img = rotate(img, angle, reshape=False, cval=0)
        
        # Scale variations
        scale = np.random.uniform(0.7, 1.3)
        h, w = img.shape
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h))
        
        if new_h > h:
            img = img[:h, :w]
        else:
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            img = cv2.copyMakeBorder(img, pad_h, h-new_h-pad_h, pad_w, w-new_w-pad_w,
                                    cv2.BORDER_CONSTANT, value=0)
        
        # Add noise
        if np.random.random() < 0.3:
            img = add_noise(img, noise_level=15)
        
        X_synthetic.append(img)
        y_synthetic.append(1)
    
    # Class 2: Triangle (Tam giác) - NHIỀU LOẠI
    print("  Generating Triangles (various types and orientations)...")
    for i in range(n_per_class):
        # Varied triangle types
        if i < n_per_class // 3:
            tri_type = 'equilateral'
        elif i < 2 * n_per_class // 3:
            tri_type = 'right'
        else:
            tri_type = 'isosceles'
        
        img = create_triangle(size=200, type=tri_type)
        
        # Random rotations (all angles)
        angle = np.random.uniform(-180, 180)
        img = rotate(img, angle, reshape=False, cval=0)
        
        # Scale variations
        scale = np.random.uniform(0.7, 1.3)
        h, w = img.shape
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h))
        
        if new_h > h:
            img = img[:h, :w]
        else:
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            img = cv2.copyMakeBorder(img, pad_h, h-new_h-pad_h, pad_w, w-new_w-pad_w,
                                    cv2.BORDER_CONSTANT, value=0)
        
        # Add noise
        if np.random.random() < 0.3:
            img = add_noise(img, noise_level=15)
        
        X_synthetic.append(img)
        y_synthetic.append(2)
    
    X_synthetic = np.array(X_synthetic)
    y_synthetic = np.array(y_synthetic)
    
    print(f"\n✅ Generated: {X_synthetic.shape}")
    print(f"   Class 0 (Circle): {np.sum(y_synthetic==0)}")
    print(f"   Class 1 (Rectangle): {np.sum(y_synthetic==1)}")
    print(f"   Class 2 (Triangle): {np.sum(y_synthetic==2)}")
    
    return X_synthetic, y_synthetic

def visualize_samples(X, y, n_samples=5):
    """Hiển thị samples"""
    fig, axes = plt.subplots(3, n_samples, figsize=(15, 9))
    
    for class_id in range(3):
        indices = np.where(y == class_id)[0]
        samples = np.random.choice(indices, n_samples, replace=False)
        
        for j, idx in enumerate(samples):
            axes[class_id, j].imshow(X[idx], cmap='gray')
            axes[class_id, j].axis('off')
            
            if j == 0:
                labels = ['Circle', 'Rectangle', 'Triangle']
                axes[class_id, j].set_ylabel(labels[class_id], 
                                             fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('synthetic_shapes_preview.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: synthetic_shapes_preview.png")

if __name__ == "__main__":
    # Generate synthetic data
    X_synthetic, y_synthetic = generate_augmented_shapes(n_per_class=500)
    
    # Visualize
    visualize_samples(X_synthetic, y_synthetic, n_samples=10)
    
    # Load original data
    print("\n📥 Loading original shapes_3classes.npz...")
    data = np.load('shapes_3classes.npz')
    X_original = data['x']
    y_original = data['y']
    print(f"   Original: {X_original.shape}")
    
    # Combine
    print("\n🔗 Combining datasets...")
    
    # Resize synthetic to match original (244, 224)
    X_synthetic_resized = np.array([cv2.resize(img, (224, 244)) 
                                    for img in X_synthetic])
    
    X_combined = np.concatenate([X_original, X_synthetic_resized], axis=0)
    y_combined = np.concatenate([y_original, y_synthetic], axis=0)
    
    print(f"✅ Combined: {X_combined.shape}")
    print(f"   Class 0: {np.sum(y_combined==0)} ({np.sum(y_combined==0)/len(y_combined)*100:.1f}%)")
    print(f"   Class 1: {np.sum(y_combined==1)} ({np.sum(y_combined==1)/len(y_combined)*100:.1f}%)")
    print(f"   Class 2: {np.sum(y_combined==2)} ({np.sum(y_combined==2)/len(y_combined)*100:.1f}%)")
    
    # Save combined dataset
    print("\n💾 Saving shapes_augmented.npz...")
    np.savez_compressed('shapes_augmented.npz', x=X_combined, y=y_combined)
    
    print("\n" + "="*80)
    print("✅ HOÀN THÀNH!")
    print("="*80)
    print("Sử dụng 'shapes_augmented.npz' thay vì 'shapes_3classes.npz' trong train_shapes.py")
    print(f"Total samples: {len(X_combined)} (Original: {len(X_original)} + Synthetic: {len(X_synthetic)})")
    print("="*80)
