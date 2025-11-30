# 🔬 BTL Xử Lý Ảnh - Nhận Dạng Chữ Số & Hình Học

Hệ thống nhận dạng chữ số viết tay (0-9) và hình học cơ bản (Tròn, Hình chữ nhật, Tam giác) sử dụng CNN và OpenCV.

## ✨ Tính năng chính

- **Nhận dạng số 0-9** với độ chính xác ~99%
- **Nhận dạng hình học** (Tròn/HCN/Tam giác) với độ chính xác ~98%
- **Pipeline xử lý ảnh 10 bước** với visualization chi tiết
- **CNN Feature Maps** - Trực quan hóa cách CNN học
- **Shape Detection** với OpenCV (Hough, Contour Analysis)
- **So sánh kỹ thuật** (Threshold, Edge Detection, Morphology)

## 📁 Cấu trúc dự án

```
BTL_XLA/
├── src/                          # Source code
│   ├── app_final.py             # Streamlit app chính
│   ├── train_model.py           # Train model số
│   ├── train_shapes.py          # Train model hình học
│   ├── config.py                # Cấu hình
│   ├── preprocessing/           # Module xử lý ảnh
│   ├── model_analysis/          # Phân tích CNN
│   ├── shape_detection/         # Phát hiện hình học
│   ├── *.h5                     # Models đã train
│   └── number_data/             # Dữ liệu training
├── Documents/                    # Tài liệu, slides
├── requirements.txt             # Dependencies
└── README.md                    # File này
```

## 🚀 Cài đặt & Chạy

### 1. Clone repository

```bash
git clone https://github.com/Thinhdc05/XLA.git
cd XLA
```

### 2. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 3. Chạy ứng dụng

```bash
cd src
streamlit run app_final.py
```

Ứng dụng sẽ mở tại: `http://localhost:8501`

## 🎯 Cách sử dụng

### Tab 1: Nhận dạng
- **Upload ảnh** hoặc **Vẽ trực tiếp** trên canvas
- Chọn mode: Nhận dạng Số (0-9) hoặc Hình học
- Click "NHẬN DẠNG" để xem kết quả

💡 **Mẹo vẽ tốt:** Độ dày nét ≥ 8px cho kết quả tốt nhất

### Tab 2: Pipeline Xử Lý
Xem 10 bước xử lý ảnh chi tiết:
1. Original → 2. Grayscale → 3. CLAHE → 4. Denoise → 5. Threshold
6. Morphology → 7. Contour → 8. Crop → 9. Resize → 10. Center

### Tab 3: Feature Maps
- Activation Maps của từng Conv layer
- Filters/Kernels CNN đã học
- Attention Heatmap (vùng quan trọng)

### Tab 4: Shape Analysis
- Phát hiện hình học tự động
- Contour analysis (Area, Perimeter, Circularity)
- Hu Moments

### Tab 5: Kỹ thuật nâng cao
So sánh các phương pháp:
- Threshold: Otsu vs Adaptive vs Binary
- Edge vs Contour Detection
- Morphology parameters

## 🛠️ Kỹ thuật áp dụng

- **Preprocessing**: CLAHE, Adaptive Threshold, Morphology
- **CNN Architecture**: 4 Conv layers (32→64→128→128 filters)
- **Shape Detection**: Hough Transform, Contour Analysis
- **Feature Extraction**: Hu Moments, Distance Transform
- **Visualization**: Feature Maps, Class Activation Map

## 📊 Kết quả

| Model | Accuracy | Dataset |
|-------|----------|---------|
| Digits (0-9) | 99.75% | Custom MNIST + 1000 ảnh |
| Shapes | 98.56% | 1000 ảnh (tròn/HCN/tam giác) |

## 📝 Training (Tùy chọn)

Nếu muốn train lại models:

```bash
cd src

# Train model nhận dạng số
python train_model.py

# Train model hình học
python train_shapes.py
```

## 🔧 Yêu cầu hệ thống

- Python 3.8+
- TensorFlow 2.x
- OpenCV 4.x
- Streamlit 1.x
- RAM: 4GB+
- Disk: 2GB

## 👨‍💻 Tác giả

Đồ án Xử lý Ảnh - K68 ĐHBK Hà Nội

## 📄 License

MIT License
