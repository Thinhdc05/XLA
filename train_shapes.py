# ==============================================================================
# TRAIN MODEL CHUYÊN NHẬN DẠNG HÌNH HỌC (Circle/Rectangle/Triangle)
# 3 classes, tập trung cao, phân biệt hình dạng tốt
# ==============================================================================

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from scipy.ndimage import center_of_mass
import glob
import os 
import zipfile 
import matplotlib.pyplot as plt
import seaborn as sns

from config import NUM_CLASSES_SHAPES, SHAPES_LABELS, MODEL_PATH_SHAPES

print("=" * 80)
print("🔺 TRAIN MODEL NHẬN DẠNG HÌNH HỌC (Circle/Rectangle/Triangle)")
print("=" * 80)

# ==============================================================================
# CẤU HÌNH
# ==============================================================================
NUM_CLASSES = NUM_CLASSES_SHAPES
EPOCHS = 35
BATCH_SIZE = 32
LEARNING_RATE = 0.001

IMAGE_DIR = 'number_data'
IMAGE_EXTENSION = '*.jpg'

SHAPE_FILES = {
    0: 'tron.new.zip',      # Circle
    1: 'hcn.new.zip',       # Rectangle
    2: 'tamgiac.new.zip'    # Triangle
}

print(f"✓ Số lớp: {NUM_CLASSES} (Circle/Rectangle/Triangle)")
print(f"✓ Epochs: {EPOCHS} | Batch: {BATCH_SIZE}")

# ==============================================================================
# TIỀN XỬ LÝ TỐI ƯU CHO HÌNH HỌC
# ==============================================================================
def preprocess_shape(img_path):
    """Tiền xử lý tối ưu cho hình học - giữ edge và contour"""
    img = cv2.imread(img_path)
    if img is None: 
        return None

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Bilateral filter - giữ edge
    bilateral = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # Adaptive threshold
    adaptive_thresh = cv2.adaptiveThreshold(
        bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, blockSize=13, C=3
    )
    
    # Morphology - kết nối contour
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    closed = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(largest)
        padding = 20  # Padding lớn hơn cho shape
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w, x + cw + padding)
        y2 = min(h, y + ch + padding)
        cropped = cleaned[y1:y2, x1:x2]
    else:
        cropped = cleaned
    
    # Resize to 20x20
    if cropped.size > 0 and cropped.shape[0] > 0 and cropped.shape[1] > 0:
        resized = cv2.resize(cropped, (20, 20), interpolation=cv2.INTER_AREA)
    else:
        resized = np.zeros((20, 20), dtype=np.uint8)
    
    # Pad to 28x28
    padded = np.pad(resized, ((4,4),(4,4)), 'constant', constant_values=0)
    
    # Center by center of mass
    if np.sum(padded) > 0:
        cy, cx = center_of_mass(padded)
        shiftx = int(np.round(14 - cx))
        shifty = int(np.round(14 - cy))
        M = np.float32([[1, 0, shiftx], [0, 1, shifty]])
        centered = cv2.warpAffine(padded, M, (28, 28))
    else:
        centered = padded
    
    # Normalize
    normalized = centered.astype(np.float32) / 255.0
    return normalized

# ==============================================================================
# LOAD DỮ LIỆU - KẾT HỢP NPZ + CUSTOM
# ==============================================================================
def load_npz_shapes():
    """Load shapes từ npz file (2080 samples)"""
    print("\n📥 Loading NPZ shape data...")
    
    if not os.path.exists('shapes_3classes.npz'):
        print("⚠️ Không tìm thấy shapes_3classes.npz")
        return None, None
    
    # Load original dataset
    data = np.load('shapes_3classes.npz')
    X_npz = data['x']  # (2080, 244, 224)
    y_npz = data['y']  # (2080,)
    
    print(f"✓ Loaded original NPZ: {X_npz.shape}")
    print(f"  Distribution: Class 0={np.sum(y_npz==0)}, Class 1={np.sum(y_npz==1)}, Class 2={np.sum(y_npz==2)}")
    
    # Load augmented dataset (with synthetic rotated rectangles)
    if os.path.exists('shapes_augmented.npz'):
        print("✓ Loading shapes_augmented.npz (synthetic data)...")
        data_aug = np.load('shapes_augmented.npz')
        X_aug = data_aug['x']  # (3580, 244, 224)
        y_aug = data_aug['y']  # (3580,)
        
        # Combine both datasets
        X_npz = np.concatenate([X_npz, X_aug[2080:]], axis=0)  # Only add synthetic part (1500 samples)
        y_npz = np.concatenate([y_npz, y_aug[2080:]], axis=0)
        
        print(f"✓ Combined with synthetic data: {X_npz.shape}")
        print(f"  Distribution: Class 0={np.sum(y_npz==0)}, Class 1={np.sum(y_npz==1)}, Class 2={np.sum(y_npz==2)}")
    else:
        print("⚠️ shapes_augmented.npz not found, using original data only")
    
    print(f"✓ Final dataset: {X_npz.shape}")
    
    # Preprocess NPZ images to 28x28
    X_processed = []
    print("  Processing NPZ images to 28x28...")
    
    for i, img in enumerate(X_npz):
        if i % 500 == 0:
            print(f"    Processed {i}/{len(X_npz)}...")
        
        # Convert PIL to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        # Preprocessing pipeline
        processed = preprocess_shape_from_array(gray)
        if processed is not None:
            X_processed.append(processed)
    
    X_processed = np.array(X_processed)
    print(f"✓ NPZ processed: {X_processed.shape}")
    
    return X_processed, y_npz

def preprocess_shape_from_array(gray):
    """Preprocess grayscale array (không cần imread)"""
    h, w = gray.shape
    
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Bilateral filter
    bilateral = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, blockSize=13, C=3
    )
    
    # Morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(largest)
        padding = 20
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w, x + cw + padding)
        y2 = min(h, y + ch + padding)
        cropped = cleaned[y1:y2, x1:x2]
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
    
    # Normalize
    normalized = centered.astype(np.float32) / 255.0
    return normalized

def load_custom_shapes():
    """Load custom shape data từ zip files (optional, ~300 samples)"""
    print("\n📥 Loading custom shape data...")
    
    X_data = []
    y_data = []
    
    for label, zip_file in SHAPE_FILES.items():
        if not os.path.exists(zip_file):
            print(f"  ⚠️ Không tìm thấy {zip_file}")
            continue
        
        # Extract
        shape_name = list(SHAPES_LABELS.values())[label]
        extract_dir = os.path.join(IMAGE_DIR, shape_name.lower().replace(' ', '_'))
        
        if not os.path.exists(extract_dir):
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(IMAGE_DIR)
        
        # Load images
        img_paths = glob.glob(os.path.join(extract_dir, IMAGE_EXTENSION))
        count = 0
        
        for img_path in img_paths:
            processed = preprocess_shape(img_path)
            if processed is not None:
                X_data.append(processed)
                y_data.append(label)
                count += 1
        
        print(f"  ✓ {shape_name}: {count} ảnh")
    
    if len(X_data) > 0:
        X_data = np.array(X_data)
        y_data = np.array(y_data)
        print(f"✓ Custom data: {X_data.shape}")
        return X_data, y_data
    else:
        return None, None

# ==============================================================================
# MAIN TRAINING
# ==============================================================================
def main():
    # Load NPZ data (primary)
    X_npz, y_npz = load_npz_shapes()
    
    # Load custom data (optional supplement)
    X_custom, y_custom = load_custom_shapes()
    
    # Combine datasets
    if X_npz is not None and X_custom is not None:
        X_data = np.concatenate([X_npz, X_custom], axis=0)
        y_data = np.concatenate([y_npz, y_custom], axis=0)
        print(f"\n✓ Combined: {X_data.shape} ({len(X_npz)} NPZ + {len(X_custom)} custom)")
    elif X_npz is not None:
        X_data = X_npz
        y_data = y_npz
        print(f"\n✓ Using NPZ only: {X_data.shape}")
    elif X_custom is not None:
        X_data = X_custom
        y_data = y_custom
        print(f"\n✓ Using custom only: {X_data.shape}")
    else:
        print("✗ Không thể train - thiếu dữ liệu!")
        return
    
    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_data, y_data, test_size=0.25, random_state=42, stratify=y_data
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.4, random_state=42, stratify=y_temp
    )
    
    # Reshape
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_val = X_val.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
    
    # To categorical
    y_train_cat = to_categorical(y_train, NUM_CLASSES)
    y_val_cat = to_categorical(y_val, NUM_CLASSES)
    y_test_cat = to_categorical(y_test, NUM_CLASSES)
    
    print(f"✓ Train: {X_train.shape}")
    print(f"✓ Val: {X_val.shape}")
    print(f"✓ Test: {X_test.shape}")
    
    # Build model - Kiến trúc khác cho shape recognition
    print("\n🏗️ Building model...")
    model = Sequential([
        Conv2D(32, (5,5), activation='relu', input_shape=(28,28,1)),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        
        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        
        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),
        
        Flatten(),
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.5),
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.4),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✓ Model compiled")
    model.summary()
    
    # Data augmentation - Thêm FLIP + tăng rotation để học hình xoay 45°
    datagen = ImageDataGenerator(
        rotation_range=40,       # Tăng 30→40° để học hình thoi (HCN xoay 45°)
        width_shift_range=0.15,  
        height_shift_range=0.15,
        zoom_range=0.15,         
        shear_range=0.1,         
        horizontal_flip=True,    # Flip ngang
        vertical_flip=True,      # Flip dọc (tam giác ngược)
        fill_mode='constant',
        cval=0
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
        ModelCheckpoint(MODEL_PATH_SHAPES, monitor='val_accuracy', save_best_only=True, verbose=1)
    ]
    
    # Train - Không dùng steps_per_epoch, để Keras tự tính từ data size
    print(f"\n🚀 Training for {EPOCHS} epochs...")
    
    history = model.fit(
        datagen.flow(X_train, y_train_cat, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(X_val, y_val_cat),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n📊 Evaluating...")
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"✓ Test Accuracy: {test_acc*100:.2f}%")
    print(f"✓ Test Loss: {test_loss:.4f}")
    
    # Confusion matrix
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=list(SHAPES_LABELS.values()),
                yticklabels=list(SHAPES_LABELS.values()))
    plt.title('Confusion Matrix - Shapes Model')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('confusion_matrix_shapes.png', dpi=150)
    print("✓ Saved: confusion_matrix_shapes.png")
    
    # Classification report
    print("\n📈 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=list(SHAPES_LABELS.values())))
    
    print("\n" + "="*80)
    print(f"✅ HOÀN THÀNH! Model đã lưu: {MODEL_PATH_SHAPES}")
    print("="*80)

if __name__ == '__main__':
    main()
