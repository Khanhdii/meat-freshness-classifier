# 🥩 Hệ thống phân loại độ tươi thịt

Ứng dụng Streamlit để phân loại độ tươi thịt sử dụng mô hình AI đã được huấn luyện.

## 📋 Tính năng

- **📤 Upload ảnh**: Upload và phân tích ảnh thịt từ thiết bị
- **📷 Chụp ảnh trực tiếp**: Sử dụng camera để chụp ảnh thịt ngay lập tức
- Phân loại thành 3 loại: **Tươi**, **Bán tươi**, **Hỏng**
- Hiển thị độ tin cậy và xác suất chi tiết cho từng lớp
- Đưa ra khuyến nghị sử dụng dựa trên kết quả
- Giao diện thân thiện với 2 tabs riêng biệt

## 🚀 Cách chạy ứng dụng

### 1. Cài đặt các thư viện cần thiết
```bash
pip install -r requirements.txt
```

### 2. Chạy ứng dụng Streamlit
```bash
streamlit run app.py
```

### 3. Mở trình duyệt
Ứng dụng sẽ tự động mở tại: `http://localhost:8501`

## 📖 Hướng dẫn sử dụng

### 📤 Tab Upload ảnh:
1. **Chọn tab "📤 Upload ảnh"**
2. **Upload ảnh**: Click vào "Browse files" để chọn ảnh thịt từ thiết bị
3. **Phân tích**: Click nút "🔍 Phân tích độ tươi (Upload)"

### 📷 Tab Chụp ảnh:
1. **Chọn tab "📷 Chụp ảnh"**
2. **Chụp ảnh**: Click vào nút camera để mở camera
3. **Chụp**: Click nút chụp để chụp ảnh thịt
4. **Phân tích**: Click nút "🔍 Phân tích độ tươi (Camera)"

### 📊 Kết quả:
Hệ thống sẽ hiển thị:
- Kết quả phân loại (Tươi/Bán tươi/Hỏng)
- Độ tin cậy của dự đoán
- Chi tiết xác suất cho từng lớp
- Khuyến nghị sử dụng cụ thể

## 🎯 Lưu ý

### 📤 Cho Upload ảnh:
- Sử dụng ảnh có độ phân giải cao để có kết quả tốt nhất
- Đảm bảo ánh sáng đủ và ảnh rõ nét
- Hỗ trợ các định dạng: PNG, JPG, JPEG

### 📷 Cho chụp ảnh trực tiếp:
- Đặt thịt trên nền sáng, phẳng
- Giữ camera ổn định khi chụp
- Đảm bảo ánh sáng đủ sáng và đều
- Chụp từ góc nhìn trực diện
- Tránh phản quang và bóng đổ che khuất

## 📁 Cấu trúc file

```
code/
├── app.py                 # Ứng dụng Streamlit chính
├── requirements.txt       # Danh sách thư viện cần thiết
├── TinyYolo_model.keras  # Model đã huấn luyện (định dạng Keras)
├── TinyYolo_model.h5     # Model đã huấn luyện (định dạng H5)
└── README.md             # File hướng dẫn này
``` 