"""
Configuration file - Cấu hình chung cho cả training và app
"""

# ==============================================================================
# MODEL CONFIG - 2 MODE SYSTEM
# ==============================================================================
INPUT_SHAPE = (28, 28, 1)

# Mode 1: Nhận dạng SỐ (0-9)
NUM_CLASSES_DIGITS = 10
DIGITS_LABELS = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
    5: '5', 6: '6', 7: '7', 8: '8', 9: '9'
}

# Mode 2: Nhận dạng HÌNH HỌC (Circle/Rectangle/Triangle)
NUM_CLASSES_SHAPES = 3
SHAPES_LABELS = {
    0: 'Tròn',
    1: 'Hình chữ nhật', 
    2: 'Tam giác'
}

# Backward compatibility (cho code cũ nếu có)
NUM_CLASSES = 13
LABELS = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
    5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'Tròn', 11: 'HCN', 12: 'Tam giác'
}

# ==============================================================================
# TRAINING CONFIG
# ==============================================================================
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

# ==============================================================================
# PREPROCESSING CONFIG
# ==============================================================================
MODE_PARAMS = {
    'Standard': {
        'block_size': 11, 'c_value': 2, 'blur': 5, 'dilation': 1, 'opening': False,
        'description': 'Tối ưu cho ảnh scan/chụp rõ nét, ánh sáng đều'
    },
    'Hand-drawn': {
        'block_size': 11, 'c_value': 2, 'blur': 7, 'dilation': 2, 
        'denoise': False, 'denoise_h': 10, 'use_otsu': True, 
        'morph_size': 3, 'close_iter': 2,
        'description': 'Tối ưu cho vẽ tay bằng bút, nét đậm'
    },
    'Photo': {
        'block_size': 15, 'c_value': 5, 'blur': 7, 'dilation': 1,
        'use_clahe': True, 'clahe_clip': 2.0, 'clahe_grid': 8,
        'use_mean': True, 'close_iter': 1,
        'description': 'Tối ưu cho ảnh chụp từ camera, ánh sáng không đều'
    },
    'Low Quality': {
        'block_size': 11, 'c_value': 3, 'blur': 5, 'dilation': 2,
        'denoise_h': 10, 'denoise_template': 7, 'denoise_search': 21,
        'clahe_clip': 3.0, 'clahe_grid': 8,
        'morph_size': 3, 'close_iter': 2,
        'description': 'Tối ưu cho ảnh mờ, tối, nhiễu'
    }
}

# ==============================================================================
# APP CONFIG - 2 MODE PATHS
# ==============================================================================
MODEL_PATH = 'best_model_production.h5'  # Legacy 13-class model
MODEL_PATH_DIGITS = 'model_digits.h5'     # Chuyên nhận dạng số
MODEL_PATH_SHAPES = 'model_shapes.h5'     # Chuyên nhận dạng hình học
