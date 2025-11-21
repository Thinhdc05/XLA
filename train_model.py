# ==============================================================================
# CODE HOÀN CHỈNH: FIXED + FULL FEATURES
# Kết hợp: Fix lỗi training + Giữ đầy đủ tính năng từ bản gốc
# Version: Production Ready
# ==============================================================================

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
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

print("=" * 80)
print("🚀 MÔ HÌNH HOÀN CHỈNH: FIXED + FULL FEATURES")
print("=" * 80)

# ==============================================================================
# I. CẤU HÌNH
# ==============================================================================
NUM_CLASSES = 13

SHAPE_LABEL_MAP = {
    'tron.new.zip': 10,
    'hcn.new.zip': 11,
    'tamgiac.new.zip': 12
}

ARCHIVE_FILES = [f'{i}.new.zip' for i in range(10)] 
ARCHIVE_FILES.extend(SHAPE_LABEL_MAP.keys()) 

IMAGE_DIR = 'number_data'
IMAGE_EXTENSION = '*.jpg' 
NPZ_FILE_NAME = 'custom_mnist_multi_13cls.npz'

TRAIN_RATIO = 0.75
VAL_RATIO = 0.15
TEST_RATIO = 0.10 
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001

print(f"✓ Số lớp: {NUM_CLASSES}")
print(f"✓ Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | LR: {LEARNING_RATE}")

# ==============================================================================
# II. TIỀN XỬ LÝ (GIỮ NGUYÊN 100%)
# ==============================================================================
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
# III. TẠO NPZ (GIỮ NGUYÊN 100%)
# ==============================================================================
def create_npz_data_multi_zip(archive_list, dest_dir, npz_path):
    """Tạo file NPZ từ các file zip"""
    if os.path.exists(npz_path):
        print(f"✓ File NPZ '{npz_path}' đã tồn tại. Bỏ qua xử lý ảnh.")
        return True
        
    print(f"\n{'='*80}")
    print(f"BƯỚC 1: XỬ LÝ ẢNH TỪ {len(archive_list)} FILE ZIP")
    print(f"{'='*80}")
    
    all_X_data = [] 
    all_y_labels = [] 
    total_files_processed = 0

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for zip_file in archive_list:
        if not os.path.exists(zip_file):
            print(f"⚠ Không tìm thấy file '{zip_file}'. Bỏ qua.")
            continue
            
        label = None
        if zip_file in SHAPE_LABEL_MAP:
            label = SHAPE_LABEL_MAP[zip_file]
        else:
            try:
                label = int(zip_file.split('.')[0])
                if label >= NUM_CLASSES:
                    raise ValueError("Nhãn vượt quá NUM_CLASSES") 
            except ValueError:
                print(f"⚠ Không xác định được nhãn từ: {zip_file}")
                continue

        if label is None or label < 0 or label >= NUM_CLASSES:
            print(f"⚠ Nhãn {label} không hợp lệ. Bỏ qua {zip_file}")
            continue
        
        print(f"→ Đang xử lý: {zip_file} (Nhãn: {label})")
        
        label_dir = os.path.join(dest_dir, str(label))
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(label_dir)
        except Exception as e:
            print(f"✗ Lỗi giải nén {zip_file}: {e}")
            continue

        label_image_files = sorted(glob.glob(os.path.join(label_dir, IMAGE_EXTENSION)))
        
        current_count = 0
        for img_file in label_image_files:
            processed_img = preprocess_to_mnist(img_file)
            
            if processed_img is not None:
                all_X_data.append(processed_img)
                all_y_labels.append(label)
                current_count += 1
                total_files_processed += 1
                if current_count == 100:
                    break
        
        print(f"   ✓ Đã xử lý {current_count} ảnh")

    if total_files_processed == 0:
        print("✗ Không có ảnh nào được xử lý thành công!")
        return False

    X_data = np.array(all_X_data, dtype=np.float32)
    y_labels = np.array(all_y_labels, dtype=np.uint8)
    X_data_3d = X_data.reshape(-1, 28, 28) 
    
    print(f"\n✓ Tổng cộng: {total_files_processed} ảnh")
    print(f"✓ Đang lưu vào '{npz_path}'...")
    
    x_test_dummy = np.empty((0, 28, 28), dtype=np.float32) 
    y_test_dummy = np.empty((0,), dtype=np.uint8)
    np.savez_compressed(
        npz_path, 
        x_train=X_data_3d, 
        y_train=y_labels, 
        x_test=x_test_dummy,
        y_test=y_test_dummy
    )
    
    print(f"✓ Lưu NPZ thành công!")
    return True

# ==============================================================================
# IV. TẢI DỮ LIỆU (GIỮ NGUYÊN 100%)
# ==============================================================================
def load_and_prepare_data(npz_path, num_classes):
    """Tải dữ liệu từ NPZ và chia Train/Val/Test"""
    print(f"\n{'='*80}")
    print("BƯỚC 2: TẢI VÀ CHUẨN BỊ DỮ LIỆU")
    print(f"{'='*80}")
    
    if not os.path.exists(npz_path): 
        return None, None, None, None, None, None
        
    data = np.load(npz_path, allow_pickle=True)
    x_full = data['x_train']
    y_full = data['y_train']
    
    x_full_cnn = x_full.reshape(x_full.shape[0], 28, 28, 1)
    y_full_ohe = to_categorical(y_full, num_classes=num_classes)
    
    if len(np.unique(y_full)) < 2:
        print("⚠ Dữ liệu chỉ có 1 nhãn")
        x_train, x_rem, y_train, y_rem = train_test_split(
            x_full_cnn, y_full_ohe, train_size=TRAIN_RATIO, random_state=42
        )
        val_size_ratio = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
        x_val, x_test, y_val, y_test = train_test_split(
            x_rem, y_rem, train_size=val_size_ratio, random_state=42
        )
    else:
        x_train, x_rem, y_train, y_rem = train_test_split(
            x_full_cnn, y_full_ohe, train_size=TRAIN_RATIO, 
            random_state=42, stratify=y_full_ohe
        )
        val_size_ratio = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
        x_val, x_test, y_val, y_test = train_test_split(
            x_rem, y_rem, train_size=val_size_ratio, 
            random_state=42, stratify=y_rem
        )

    print(f"✓ Tổng: {x_full_cnn.shape[0]} | Train: {x_train.shape[0]} | Val: {x_val.shape[0]} | Test: {x_test.shape[0]}")
    return x_train, y_train, x_val, y_val, x_test, y_test

# ==============================================================================
# V. 🔧 DATA AUGMENTATION FIXED (Nhẹ nhàng)
# ==============================================================================
def create_fixed_augmentation():
    """
    🔧 FIXED: Augmentation nhẹ nhàng cho ảnh 28×28
    - Rotation: ±15° (vừa đủ cho góc nghiêng tự nhiên)
    - Shift: ±5% (giữ đối tượng trong frame)
    - Zoom: 95-105% (tỷ lệ hợp lý)
    - KHÔNG dùng brightness (tránh mất ảnh binary)
    """
    print(f"\n{'='*80}")
    print("🔧 DATA AUGMENTATION FIXED - Nhẹ nhàng & Hiệu quả")
    print(f"{'='*80}")
    
    datagen = ImageDataGenerator(
        rotation_range=15,          # ±15° (giảm từ 30°)
        width_shift_range=0.05,     # ±5% (giảm từ 10%)
        height_shift_range=0.05,    # ±5%
        zoom_range=0.05,            # 95-105% (giảm từ 10%)
        fill_mode='constant',
        cval=0.0,
        horizontal_flip=False,
        vertical_flip=False
    )
    
    print("✅ Rotation: ±15° | Shift: ±5% | Zoom: 95-105%")
    print("✅ BỎ: Brightness, Shear (tránh mất thông tin)")
    print("✅ Phù hợp: Ảnh 28×28, binary images")
    
    return datagen

# ==============================================================================
# VI. 🔧 MÔ HÌNH FIXED (Đơn giản & Hiệu quả)
# ==============================================================================
def build_fixed_lenet(input_shape, num_classes):
    """
    🔧 FIXED: LeNet đơn giản & hiệu quả
    - Dùng Flatten (giữ spatial info)
    - Giảm Dropout: 0.2 → 0.3 → 0.5
    - Giảm L2 Reg: 0.00005
    - BỎ Spatial Dropout (quá mạnh)
    """
    print(f"\n{'='*80}")
    print("🔧 MÔ HÌNH FIXED - Đơn giản & Hiệu quả")
    print(f"{'='*80}")
    
    model = Sequential([
        # Block 1
        Conv2D(32, (5, 5), activation='relu', padding='same',
               input_shape=input_shape, 
               kernel_regularizer=l2(0.00005), name='C1'),
        BatchNormalization(name='BN1'),
        MaxPooling2D((2, 2), name='P1'),
        Dropout(0.2, name='Drop1'),
        
        # Block 2
        Conv2D(64, (5, 5), activation='relu', padding='same',
               kernel_regularizer=l2(0.00005), name='C2'),
        BatchNormalization(name='BN2'),
        MaxPooling2D((2, 2), name='P2'),
        Dropout(0.3, name='Drop2'),
        
        # Block 3
        Conv2D(128, (3, 3), activation='relu', padding='same',
               kernel_regularizer=l2(0.00005), name='C3'),
        BatchNormalization(name='BN3'),
        Dropout(0.3, name='Drop3'),
        
        # Flatten (thay vì Global Avg Pool)
        Flatten(name='Flatten'),
        
        # Fully Connected
        Dense(256, activation='relu', 
              kernel_regularizer=l2(0.00005), name='FC1'),
        BatchNormalization(name='BN4'),
        Dropout(0.5, name='Drop4'),
        
        Dense(num_classes, activation='softmax', name='Output')
    ])
    
    model.summary()
    print("\n" + "="*80)
    print("✨ CÁC ĐIỂM MẠNH:")
    print("="*80)
    print("✅ Flatten: Giữ đầy đủ spatial information")
    print("✅ Dropout tăng dần: 0.2 → 0.3 → 0.5")
    print("✅ L2 Reg nhẹ: 0.00005 (không quá aggressive)")
    print("✅ BatchNorm: Training ổn định")
    print(f"✅ Tổng tham số: {model.count_params():,}")
    print("="*80)
    
    return model

# ==============================================================================
# VII. TRAINING VỚI AUGMENTATION
# ==============================================================================
def train_model(model, datagen, x_train, y_train, x_val, y_val, 
                x_test, y_test, epochs, batch_size, lr):
    """Huấn luyện mô hình với callbacks đầy đủ"""
    print(f"\n{'='*80}")
    print("🚀 BẮT ĐẦU TRAINING")
    print(f"{'='*80}")
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer, 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )

    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=15,
        restore_best_weights=True, 
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=7,
        min_lr=1e-7,
        verbose=1
    )
    
    checkpoint = ModelCheckpoint(
        'best_model_production.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    # Fit với augmentation
    datagen.fit(x_train)
    
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(x_train) // batch_size,
        validation_data=(x_val, y_val),
        epochs=epochs,
        callbacks=[early_stop, reduce_lr, checkpoint],
        verbose=1
    )

    # Đánh giá Test
    if x_test.shape[0] > 0:
        print(f"\n{'='*80}")
        print("📊 ĐÁNH GIÁ TEST SET")
        print(f"{'='*80}")
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        print(f"✅ Test Accuracy: {accuracy*100:.2f}%")
        print(f"✅ Test Loss: {loss:.4f}")
        
        if accuracy >= 0.95:
            print("🎉 XUẤT SẮC! Accuracy >= 95%")
        elif accuracy >= 0.85:
            print("👍 TỐT! Accuracy >= 85%")
        elif accuracy >= 0.70:
            print("⚠️ Chấp nhận được. Có thể cải thiện")
        else:
            print("❌ Cần cải thiện thêm")
    
    return history

# ==============================================================================
# VIII. 🆕 TEST TIME AUGMENTATION (TTA)
# ==============================================================================
def predict_with_tta(model, image, datagen, n_augment=10):
    """
    Test Time Augmentation: Dự đoán trên nhiều phiên bản augmented
    Tăng độ chính xác 2-5% trên ảnh khó
    """
    predictions = []
    
    # Dự đoán ảnh gốc
    pred = model.predict(image[np.newaxis, ...], verbose=0)
    predictions.append(pred)
    
    # Dự đoán các ảnh augmented
    for _ in range(n_augment - 1):
        aug_img = datagen.random_transform(image)
        pred = model.predict(aug_img[np.newaxis, ...], verbose=0)
        predictions.append(pred)
    
    # Trung bình predictions
    avg_pred = np.mean(predictions, axis=0)
    return avg_pred

# ==============================================================================
# IX. 🆕 HÀM TIỆN ÍCH - DỰ ĐOÁN ẢNH MỚI
# ==============================================================================
def predict_single_image(model, img_path, use_tta=False, datagen=None):
    """
    Hàm tiện ích dự đoán 1 ảnh mới
    
    Args:
        model: Mô hình đã train
        img_path: Đường dẫn ảnh hoặc numpy array
        use_tta: Có dùng Test Time Augmentation không
        datagen: ImageDataGenerator (cần nếu use_tta=True)
    
    Returns:
        label: Nhãn dự đoán
        confidence: Độ tin cậy (%)
        preprocessed_img: Ảnh sau preprocessing (để debug)
    """
    # Tiền xử lý
    if isinstance(img_path, str):
        img = preprocess_to_mnist(img_path)
        if img is None:
            return None, 0.0, None
    else:
        img = img_path
    
    img_input = img.reshape(1, 28, 28, 1)
    
    # Dự đoán
    if use_tta and datagen is not None:
        pred = predict_with_tta(model, img_input[0], datagen, n_augment=10)
    else:
        pred = model.predict(img_input, verbose=0)
    
    class_idx = np.argmax(pred)
    confidence = pred[0][class_idx] * 100
    
    # Map label
    label_names = {i: str(i) for i in range(10)}
    label_names.update({10: 'Tròn', 11: 'HCN', 12: 'Tam giác'})
    
    return label_names[class_idx], confidence, img

# ==============================================================================
# X. 🆕 VISUALIZE PREPROCESSING PIPELINE
# ==============================================================================
def visualize_preprocessing(img_path):
    """Hiển thị từng bước preprocessing"""
    print(f"\n{'='*80}")
    print("🔍 PIPELINE PREPROCESSING")
    print(f"{'='*80}")
    
    # Đọc ảnh
    img = cv2.imread(img_path)
    if img is None:
        print("❌ Không đọc được ảnh!")
        return
    
    # Các bước xử lý
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, blockSize=9, C=2
    )
    final = preprocess_to_mnist(img_path)
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('1. Ảnh gốc', fontsize=12, weight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gray, cmap='gray')
    axes[0, 1].set_title('2. Grayscale', fontsize=12)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(blurred, cmap='gray')
    axes[0, 2].set_title('3. Gaussian Blur', fontsize=12)
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(adaptive_thresh, cmap='gray')
    axes[1, 0].set_title('4. Adaptive Threshold', fontsize=12)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(final, cmap='gray')
    axes[1, 1].set_title('5. Final (28×28)', fontsize=12, weight='bold', color='green')
    axes[1, 1].axis('off')
    
    # Thống kê
    axes[1, 2].axis('off')
    stats_text = f"""
    📊 THỐNG KÊ:
    
    • Shape cuối: {final.shape}
    • Min pixel: {final.min():.3f}
    • Max pixel: {final.max():.3f}
    • Mean: {final.mean():.3f}
    • Std: {final.std():.3f}
    • White pixels: {np.sum(final > 0.5)}
    • Coverage: {np.sum(final > 0.5)/784*100:.1f}%
    """
    axes[1, 2].text(0.1, 0.5, stats_text, fontsize=11, 
                   verticalalignment='center', family='monospace')
    
    plt.tight_layout()
    plt.suptitle('Pipeline Preprocessing - Từng Bước Chi Tiết', 
                y=1.02, fontsize=16, weight='bold')
    plt.show()

# ==============================================================================
# XI. THỰC THI CHÍNH
# ==============================================================================
print(f"\n{'='*80}")
print("▶️ BẮT ĐẦU QUY TRÌNH")
print(f"{'='*80}")

# 1. Tạo NPZ
if not create_npz_data_multi_zip(ARCHIVE_FILES, IMAGE_DIR, NPZ_FILE_NAME):
    raise FileNotFoundError(f"Không thể tạo file NPZ: {NPZ_FILE_NAME}")

# 2. Tải dữ liệu
x_train, y_train, x_val, y_val, x_test, y_test = load_and_prepare_data(
    NPZ_FILE_NAME, NUM_CLASSES
)

if x_train is None or x_train.shape[0] == 0:
    raise ValueError("Dữ liệu không hợp lệ!")

INPUT_SHAPE = x_train.shape[1:]
print(f"\n✓ Input Shape: {INPUT_SHAPE}")

# 3. Tạo Data Augmentation
datagen = create_fixed_augmentation()

# ==============================================================================
# XII. VISUALIZE AUGMENTED DATA
# ==============================================================================
print(f"\n{'='*80}")
print("🖼️ TRỰC QUAN HÓA DATA AUGMENTATION")
print(f"{'='*80}")

if x_train.shape[0] > 0:
    sample_img = x_train[0:1]
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    # Ảnh gốc
    axes[0].imshow(sample_img[0].squeeze(), cmap='gray')
    axes[0].set_title('🔷 Ảnh gốc', fontsize=12, color='green', weight='bold')
    axes[0].axis('off')
    
    # 9 ảnh augmented
    datagen.fit(sample_img)
    aug_iter = datagen.flow(sample_img, batch_size=1)
    
    for i in range(1, 10):
        aug_img = next(aug_iter)[0]
        axes[i].imshow(aug_img.squeeze(), cmap='gray')
        axes[i].set_title(f'Augmented {i}', fontsize=11)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.suptitle('Augmentation FIXED: Nhẹ nhàng, giữ ảnh trong frame', 
                 y=1.02, fontsize=16, weight='bold')
    plt.show()

# ==============================================================================
# XIII. XÂY DỰNG VÀ HUẤN LUYỆN MÔ HÌNH
# ==============================================================================
model = build_fixed_lenet(INPUT_SHAPE, NUM_CLASSES)

history = train_model(
    model, datagen,
    x_train, y_train, x_val, y_val, x_test, y_test,
    EPOCHS, BATCH_SIZE, LEARNING_RATE
)

# Lưu model cuối cùng
model.save('lenet_production_final.h5')
print("\n✅ Đã lưu: lenet_production_final.h5")
print("✅ Best model: best_model_production.h5")

# ==============================================================================
# XIV. CONFUSION MATRIX & CLASSIFICATION REPORT
# ==============================================================================
print(f"\n{'='*80}")
print("📈 CONFUSION MATRIX & CLASSIFICATION REPORT")
print(f"{'='*80}")

if x_test.shape[0] > 0:
    y_pred = model.predict(x_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    # Confusion Matrix với màu đẹp
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', 
                xticklabels=range(NUM_CLASSES), 
                yticklabels=range(NUM_CLASSES),
                cbar_kws={'label': 'Số lượng dự đoán'})
    plt.title('Confusion Matrix - Model Production', fontsize=16, weight='bold')
    plt.ylabel('Nhãn thực tế', fontsize=13)
    plt.xlabel('Nhãn dự đoán', fontsize=13)
    plt.show()
    
    # Classification Report
    print("\n📋 Classification Report:")
    target_names = [str(i) for i in range(10)] + ['Tròn (10)', 'HCN (11)', 'Tam giác (12)']
    print(classification_report(y_true_classes, y_pred_classes, 
                                target_names=target_names, zero_division=0))
    
    # Phân tích classes
    diagonal = np.diag(cm)
    class_totals = cm.sum(axis=1)
    class_acc = diagonal / class_totals
    
    print(f"\n🔍 Phân tích từng Class:")
    for i in range(NUM_CLASSES):
        status = "🎉" if class_acc[i] >= 0.95 else "👍" if class_acc[i] >= 0.80 else "⚠️"
        print(f"   {status} Class {i}: {class_acc[i]*100:.1f}% ({diagonal[i]}/{class_totals[i]})")

# ==============================================================================
# XV. DỰ ĐOÁN VÀ VISUALIZE KÈM CONFIDENCE
# ==============================================================================
print(f"\n{'='*80}")
print("🎯 DỰ ĐOÁN TRÊN TẬP TEST (Kèm Confidence)")
print(f"{'='*80}")

if x_test.shape[0] > 0:
    n_samples = min(x_test.shape[0], 10)
    predictions = model.predict(x_test[:n_samples], verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test[:n_samples], axis=1)
    
    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    axes = axes.flatten()
    
    label_names = {i: str(i) for i in range(10)}
    label_names.update({10: 'Tròn', 11: 'HCN', 12: 'Tam giác'})
    
    for i in range(n_samples):
        img = x_test[i].squeeze()
        is_correct = (predicted_classes[i] == true_classes[i])
        
        true_name = label_names[true_classes[i]]
        pred_name = label_names[predicted_classes[i]]
        confidence = predictions[i][predicted_classes[i]] * 100
        
        # Màu sắc theo độ tin cậy
        if is_correct:
            if confidence >= 95:
                color = 'darkgreen'
            else:
                color = 'green'
        else:
            color = 'red'
        
        title = f"T:{true_name} | P:{pred_name}\nConfidence: {confidence:.1f}%"
        
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(title, color=color, fontsize=10, weight='bold')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.suptitle('Dự đoán với Model Production (Top 10 mẫu)', 
                 y=1.02, fontsize=16, weight='bold')
    plt.show()

# ==============================================================================
# XVI. TRAINING HISTORY VISUALIZATION
# ==============================================================================
print(f"\n{'='*80}")
print("📊 BIỂU ĐỒ TRAINING HISTORY")
print(f"{'='*80}")

if history is not None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy Plot
    axes[0].plot(history.history['accuracy'], label='Train Acc', 
                linewidth=2.5, marker='o', markersize=4, color='#2E86AB')
    axes[0].plot(history.history['val_accuracy'], label='Val Acc', 
                linewidth=2.5, marker='s', markersize=4, color='#A23B72')
    axes[0].set_title('Accuracy qua các Epoch', fontsize=14, weight='bold')
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].legend(fontsize=11, loc='lower right')
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].set_ylim([0.5, 1.05])
    
    # Loss Plot
    axes[1].plot(history.history['loss'], label='Train Loss', 
                linewidth=2.5, marker='o', markersize=4, color='#F18F01')
    axes[1].plot(history.history['val_loss'], label='Val Loss', 
                linewidth=2.5, marker='s', markersize=4, color='#C73E1D')
    axes[1].set_title('Loss qua các Epoch', fontsize=14, weight='bold')
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].legend(fontsize=11, loc='upper right')
    axes[1].grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.show()
    
    # In thống kê
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    best_val_acc = max(history.history['val_accuracy'])
    best_epoch = np.argmax(history.history['val_accuracy']) + 1
    
    print(f"\n📈 THỐNG KÊ TRAINING:")
    print(f"   • Final Train Accuracy: {final_train_acc*100:.2f}%")
    print(f"   • Final Val Accuracy: {final_val_acc*100:.2f}%")
    print(f"   • Best Val Accuracy: {best_val_acc*100:.2f}% (Epoch {best_epoch})")
    print(f"   • Tổng Epochs đã chạy: {len(history.history['accuracy'])}")
    
    # Kiểm tra overfitting
    gap = final_train_acc - final_val_acc
    print(f"\n🔍 Kiểm tra Overfitting:")
    print(f"   • Train-Val Gap: {gap*100:.2f}%")
    if gap > 0.10:
        print("   ⚠️ Overfitting nhẹ (gap > 10%)")
    elif gap < 0.05:
        print("   ⚠️ Có thể underfitting (gap < 5%)")
    else:
        print("   ✅ Cân bằng tốt!")

# ==============================================================================
# XVII. 🆕 TEST TIME AUGMENTATION (TTA) - DEMO
# ==============================================================================
print(f"\n{'='*80}")
print("🔬 TEST TIME AUGMENTATION (TTA) - SO SÁNH")
print(f"{'='*80}")

if x_test.shape[0] > 0:
    # Chọn 5 mẫu test
    n_samples = min(5, x_test.shape[0])
    
    print("\n📊 So sánh Normal vs TTA trên 5 mẫu:")
    print("-" * 80)
    
    for i in range(n_samples):
        test_sample = x_test[i]
        true_label = np.argmax(y_test[i])
        
        # Dự đoán thông thường
        normal_pred = model.predict(test_sample[np.newaxis, ...], verbose=0)
        normal_class = np.argmax(normal_pred)
        normal_conf = normal_pred[0][normal_class] * 100
        
        # Dự đoán với TTA
        tta_pred = predict_with_tta(model, test_sample, datagen, n_augment=10)
        tta_class = np.argmax(tta_pred)
        tta_conf = tta_pred[0][tta_class] * 100
        
        label_names = {i: str(i) for i in range(10)}
        label_names.update({10: 'Tròn', 11: 'HCN', 12: 'Tam giác'})
        
        improvement = tta_conf - normal_conf
        arrow = "📈" if improvement > 0 else "📉"
        
        print(f"Mẫu {i+1} - True: {label_names[true_label]}")
        print(f"  • Normal: {label_names[normal_class]} ({normal_conf:.2f}%)")
        print(f"  • TTA:    {label_names[tta_class]} ({tta_conf:.2f}%)")
        print(f"  • {arrow} Cải thiện: {improvement:+.2f}%")
        print()


