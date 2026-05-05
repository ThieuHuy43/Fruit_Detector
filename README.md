# Ứng Dụng Nhận Diện Nông Sản & Phát Hiện Hàng Giả (TF-MCA / BiMC / CLIP)

Đây là hệ thống phân loại và xác thực nguồn gốc nông sản bằng Trí tuệ Nhân tạo đa phương thức (Multimodal AI), kết hợp thuật toán **TF-MCA** (Training-Free Memory-based Class-incremental Adaptation) và **CLIP** để đạt được khả năng **Few-shot Class-Incremental** (nhận biết nông sản mới qua vài bức ảnh mà không cần chạy huấn luyện lại mô hình).

Dự án bao gồm 2 phần chính:
1. **Backend (FastAPI, PyTorch):** Xử lý luồng AI lõi.
2. **Frontend (Flutter):** Giao diện trên thiết bị di động (Android / iOS / Web).

---

## 🚀 1. Các Tính Năng Nổi Bật

- **Nhận diện chính xác:** Sử dụng mô hình `ViT-B/16` của OpenAI CLIP kết hợp với không gian đặc trưng Fused Prototypes (nhúng ảnh và chữ cùng nhau).
- **Cảnh báo hàng giả (OOD Detection):** Tính toán **Khoảng cách Mahalanobis** (Khoảng cách Covariance) để đánh giá độ lệch chuẩn. Hình ảnh không thuộc phân phối của lớp chuẩn sẽ bị cảnh báo "Giả/Không đạt chuẩn".
- **Học tăng cường không cần huấn luyện lại (Training-Free Incremental):** Add Class (Thêm Nông sản) bằng cách lấy `mean` đặc trưng các ảnh mẫu (Prototypes) và băm vào RAM (`bimc_memory.pth`).
- **Dịch thuật song ngữ tự động:** Dịch nhãn lớp từ logic (VD: `mango-DongThap`) sang ngôn ngữ giao diện (VD: `Xoài (Đồng Tháp)`).

---

## 🧠 2. Cách Chức Hệ Thống Hoạt Động (AI Logic)

1. **Khởi tạo:** Hệ thống nạp trọng số CLIP và đọc file trí nhớ `bimc_memory.pth` chứa sẵn các Prototypes và tham số thích ứng `beta`, `lambda_t`.
2. **Phân loại ảnh (`/predict`):** 
   - Mã hóa hình ảnh qua CLIP (trích xuất đặc trưng).
   - Đo độ tương đồng *Cosine Similarity* giữa véc-tơ ảnh và các cụm `fused_proto` từ bộ nhớ để tìm nhãn tốt nhất.
   - Tính toán Mahalanobis Distance để sinh ra mức `fake_score`. Kết quả ensemble trả về cho ứng dụng.
3. **Thêm lớp mới (`/add_class`):** 
   - Đọc 1-N hình ảnh cùng 1 đoạn "Mô tả đặc trưng" (tuỳ chọn) làm Text Prototype.
   - Chạy xuôi (Forward pass) mô hình CLIP để tính các nguyên mẫu chuẩn hóa mà không cập nhật đạo hàm (no gradient update).
   - Nối đặc trưng mới vào biến cấu trúc trong RAM và ghi đè `bimc_memory.pth` để bảo lưu độ hiểu biết.

---

## ⚙️ 3. Hướng Dẫn Cài Đặt (Local Environment)

### A. Cài đặt Backend (Python)
Yêu cầu: `Python 3.9+`, `CUDA` (Nếu chạy GPU).

1. Mở Terminal / CMD, trỏ vào thư mục `backend/`
2. Cài đặt các thư viện lõi:
   ```bash
   pip install fastapi uvicorn python-multipart torch torchvision Pillow numpy
   ```
   *(lưu ý: Hãy tải pytorch phiên bản có CUDA tương ứng tại `pytorch.org` để mô hình chạy với tốc độ cao nhất).*
3. Chạy Server AI:
   ```bash
   cd backend
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```
   *(Server khởi động xong sẽ hiện thông báo `Loaded X classes từ BiMC memory`)*

### B. Cài đặt Frontend (Flutter)
Yêu cầu: `Flutter SDK 3.x+` đã hoạt động.

1. Khởi động máy ảo Android/iOS hoặc cắm thiết bị thật. Nếu muốn nhanh có thể build ra Chrome (`Web`).
2. Trỏ Terminal vào thư mục ứng dụng Flutter (Ví dụ `app/`).
3. Khôi phục thư viện Dart:
   ```bash
   flutter pub get
   ```
4. Kiểm tra cấu hình IP trong file kết nối API (thường ở file `api_service.dart`):
   - Đổi dòng `const String baseUrl = 'http://127.0.0.1:8000';` 
   - Nếu chạy qua Điện thoại vật lý, đổi `127.0.0.1` thành **IP WLAN** của máy tính chạy backend mạng nội bộ (VD: `192.168.1.5`).
5. Chạy ứng dụng:
   ```bash
   flutter run -d chrome
   ```

---

## 🛠️ Trợ Giúp / Nâng Cấp Hệ Thống
- **Xóa / Reset Trí nhớ:** Chỉ cần xóa hoặc chuyển tệp `bimc_memory.pth` đi chỗ khác, hệ thống sẽ cảnh báo không có dữ liệu để dự đoán.
- **Biên dịch từ mới:** Sửa đổi bộ từ điển song ngữ `FRUIT_TRANSLATION` và `ORIGIN_TRANSLATION` trực tiếp tại file `backend/model/tf_mca.py`.
