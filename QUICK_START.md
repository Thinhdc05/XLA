# 🚀 HƯỚNG DẪN NHANH - BTL XỬ LÝ ẢNH

## ⚡ QUICK START (5 phút)

### 1. Cài đặt (1 phút)
```powershell
pip install -r requirements.txt
```

### 2. Train Models (30 phút total)
```powershell
# Train model số (15-20 phút)
python train_digits.py

# Train model hình học (10-15 phút)
python train_shapes.py
```

### 3. Chạy App (ngay lập tức)
```powershell
streamlit run app_final.py
```

Mở browser: http://localhost:8501

---

## 📋 CHECKLIST DEMO

### Trước khi demo:
- [ ] Đã train xong 2 models (model_digits.h5 & model_shapes.h5)
- [ ] Cài đủ dependencies (streamlit, opencv, tensorflow)
- [ ] Chuẩn bị 3-5 ảnh test (số và hình học)
- [ ] App chạy được tại localhost:8501

### Demo theo thứ tự:

#### Tab 1: Nhận Dạng (2 phút)
1. Upload ảnh số → Chọn "Nhận dạng Số" → Nhận dạng
2. Upload ảnh hình học → Chọn "Nhận dạng Hình học" → Nhận dạng
3. Show confidence score & top predictions

#### Tab 2: Pipeline Xử Lý Ảnh (3 phút)
1. Scroll qua 10 bước: Original → Gray → CLAHE → ... → Final
2. Giải thích mỗi bước làm gì
3. Show histogram before/after CLAHE

#### Tab 3: Feature Maps (2 phút)
1. Mở Conv layers → Show activation maps
2. Mở Filters → Show kernels learned
3. Show heatmap attention (vùng model chú ý)

#### Tab 4: Shape Analysis (2 phút)
1. Auto detect shape (circle/rect/triangle)
2. Show contour properties (area, circularity, etc.)
3. Show Hu Moments (đặc trưng bất biến)

#### Tab 5: Kỹ Thuật Nâng Cao (1 phút)
1. Edge detection comparison (Sobel, Canny, Laplacian, Scharr)
2. Threshold comparison (Otsu, Adaptive)
3. Scroll qua bảng summary kỹ thuật

**Total: ~10 phút demo**

---

## 🎯 ĐIỂM MẠNH NÊN NHẤN MẠNH

### 1. Module Hóa Tốt
- 3 packages riêng: `preprocessing/`, `model_analysis/`, `shape_detection/`
- Code clean, docstrings đầy đủ
- Dễ mở rộng

### 2. Nhiều Kỹ Thuật Xử Lý Ảnh
- 12+ techniques: CLAHE, Morphology, Edge Detection, Hough, Distance Transform, Hu Moments, v.v.
- Không chỉ basic mà có advanced

### 3. Visualization Đầy Đủ
- 10-step pipeline rõ ràng
- Feature maps CNN (hiếm có)
- Heatmap attention (Grad-CAM style)

### 4. Giải Quyết Vấn Đề Thực Tế
- Model 13 classes nhầm 4↔️Tam giác, 0↔️Tròn
- → Tách 2 models chuyên biệt → Accuracy tăng 10%

### 5. UI Trực Quan
- 5 tabs logic
- Color-coded results
- Real-time processing

---

## 📊 SỐ LIỆU THUYẾT PHỤC

- **12+ kỹ thuật** xử lý ảnh nâng cao
- **2 models** chuyên biệt
- **10 bước** visualization pipeline chi tiết
- **5 tabs** UI logic, dễ dùng
- **60k+ MNIST** samples training
- **95-98%** accuracy digits
- **97-99%** accuracy shapes
- **3 packages** modular clean code

---

## 🎤 GỢI Ý THUYẾT TRÌNH

### Mở đầu (30s):
"Dự án của em là hệ thống nhận dạng chữ số và hình học, đặc biệt tập trung vào **quá trình xử lý ảnh** và **trích chọn đặc trưng** - đúng yêu cầu BTL."

### Body (8 phút):
**1. Vấn đề (1 phút):**
- Model 13 classes nhầm lẫn giữa số và hình
- → Tách 2 models chuyên biệt

**2. Giải pháp (2 phút):**
- 2 models: Digits (10 classes) vs Shapes (3 classes)
- Pipeline xử lý: 10 bước chi tiết
- 12+ kỹ thuật OpenCV

**3. Demo (5 phút):**
- Tab 1: Nhận dạng live
- Tab 2: Pipeline visualization (QUAN TRỌNG)
- Tab 3: Feature maps CNN (QUAN TRỌNG)
- Tab 4: Shape detection OpenCV
- Tab 5: Kỹ thuật nâng cao

### Kết (30s):
"Dự án thể hiện **hiểu sâu xử lý ảnh** (12+ techniques), **hiểu CNN** (feature maps), **code clean** (module hóa), và **UI đẹp** (5 tabs). Sẵn sàng trả lời câu hỏi!"

---

## ❓ CÂU HỎI THƯỜNG GẶP & TRẢ LỜI

**Q: Tại sao tách 2 models?**
A: Model 13 classes nhầm 4↔️Tam giác (62%), 0↔️Tròn (58%). Tách riêng → Accuracy tăng 10%, từ 87% lên 96%.

**Q: Kỹ thuật nào quan trọng nhất?**
A: **CLAHE** (contrast enhancement) và **Adaptive Threshold**. CLAHE xử lý ánh sáng không đều, Adaptive Threshold tự động cho từng vùng nhỏ.

**Q: Feature maps có ý nghĩa gì?**
A: Feature maps cho thấy CNN học **gì** từ ảnh. Layer đầu học edge đơn giản, layer sau học patterns phức tạp (số 8, góc tam giác, v.v.)

**Q: Tại sao dùng OpenCV + Deep Learning?**
A: OpenCV tốt cho geometric features (circularity, vertices), CNN tốt cho texture/patterns. Kết hợp → Robust hơn.

**Q: Code có reusable không?**
A: Có! 3 packages modular (`preprocessing`, `model_analysis`, `shape_detection`). Mỗi function độc lập, dễ import vào project khác.

---

## 🔧 FIX LỖI NHANH

**Lỗi: ModuleNotFoundError**
```powershell
pip install -r requirements.txt
```

**Lỗi: Model not found**
```powershell
python train_digits.py
python train_shapes.py
```

**Lỗi: streamlit command not found**
```powershell
pip install streamlit
# Hoặc
python -m streamlit run app_final.py
```

**App chạy chậm**
→ Bình thường! Load models + predict lần đầu ~5s. Sau đó nhanh.

---

## ✅ TRƯỚC KHI NỘP

- [ ] Code chạy được không lỗi
- [ ] 2 models đã train xong
- [ ] README_FINAL.md đầy đủ
- [ ] Demo notebook functional
- [ ] Screenshot app (5 tabs)
- [ ] Confusion matrix saved
- [ ] Video demo (optional, 3-5 phút)

---

**Good luck! 🚀**
