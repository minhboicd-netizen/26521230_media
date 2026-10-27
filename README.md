# Ứng dụng Bảng vẽ trên không với MediaPipe

Đây là bài tập lớn (Assignment) xây dựng ứng dụng tương tác với Camera sử dụng ngôn ngữ **Python**, thư viện **OpenCV** và **MediaPipe**. Chương trình cho phép người dùng sử dụng ngón tay di chuyển trước Webcam để vẽ tranh, đổi màu bút và xóa bảng trực tiếp trên màn hình.

## 1. Ý tưởng và quy trình hoạt động

Thay vì dùng chuột máy tính, người dùng đưa bàn tay lên trước camera để điều khiển con trỏ vẽ. Quy trình xử lý của chương trình đi theo các bước:

`Hình ảnh từ Camera` -> `MediaPipe (lấy 21 điểm khớp bàn tay)` -> `Thuật toán kiểm tra ngón tay duỗi hay gập` -> `Chuyển đổi sang chế độ Vẽ / Chọn màu / Xóa bảng` -> `Hiển thị nét vẽ lên màn hình`

### Các thao tác cử chỉ trong ứng dụng:
- **Vẽ nét (Giơ 1 ngón trỏ)**: Khi chỉ có ngón trỏ duỗi thẳng, chương trình lấy tọa độ đầu ngón trỏ (điểm số 8) để vẽ nét mực lên màn hình.
- **Di chuyển và Chọn công cụ (Giơ 2 ngón trỏ và giữa)**: Khi giơ 2 ngón tay, bút vẽ sẽ tạm ngưng để người dùng di chuyển tay mà không bị lem nét. Nếu đưa 2 ngón tay chạm vào thanh công cụ ở cạnh trên màn hình, người dùng có thể đổi màu bút (Xanh dương, Xanh lá, Đỏ, Vàng), chọn **Cục tẩy** hoặc bấm **Xóa hết**.
- **Tạm ngừng vẽ (Nắm bàn tay)**: Khi nắm tay lại, chương trình ngắt nét vẽ hiện tại để bạn chuyển sang vị trí khác trên màn hình.
- **Xóa nhanh toàn bộ bảng (Xòe cả 5 ngón tay)**: Giữ bàn tay xòe đủ 5 ngón trong khoảng 1 giây để xóa sạch toàn bộ nét vẽ.

## 2. Các thư viện cần thiết

Chương trình tương thích với cả **Python 3.9 – 3.14** (hỗ trợ cả chuẩn `HandLandmarker` trên MediaPipe 1.0+ lẫn `mp.solutions.hands` trên các bản cũ) và sử dụng 3 thư viện chính trong file `requirements.txt`:
- `opencv-python`: Đọc hình ảnh từ camera, vẽ giao diện thanh công cụ và ghép lớp nét vẽ.
- `mediapipe`: Nhận diện bàn tay và trích xuất tọa độ 21 điểm khớp xương.
- `numpy`: Tạo mảng ma trận lưu lớp mực vẽ.

## 3. Hướng dẫn cài đặt và chạy chương trình

### Bước 1: Tải mã nguồn về máy
```bash
git clone https://github.com/<ten-tai-khoan-cua-ban>/bang-ve-mediapipe.git
cd bang-ve-mediapipe
```

### Bước 2: Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### Bước 3: Chạy ứng dụng
```bash
# Trên Windows:
py main.py
# Hoặc:
python main.py
```
*(Lưu ý: Ở lần chạy đầu tiên với MediaPipe 1.0+, chương trình sẽ tự động tải file mô hình `hand_landmarker.task` của Google về thư mục dự án).*

## 4. Phím tắt hỗ trợ trên bàn phím

Ngoài việc điều khiển bằng cử chỉ tay, bạn có thể dùng thêm các phím tắt sau (nhớ bấm chuột vào cửa sổ Camera trước khi nhấn phím):
- Phím **`[+]`** hoặc **`[-]`**: Tăng hoặc giảm độ dày nét bút.
- Phím **`[C]`**: Xóa sạch bảng vẽ ngay lập tức.
- Phím **`[S]`**: Lưu bức tranh đang vẽ thành file ảnh `.png`.
- Phím **`[Q]`** hoặc **`[ESC]`**: Thoát chương trình.

## 5. Cấu trúc thư mục

| Tên file | Nội dung |
| :--- | :--- |
| `main.py` | Toàn bộ mã nguồn xử lý camera, thuật toán kiểm tra ngón tay và giao diện bảng vẽ |
| `requirements.txt` | Danh sách thư viện cần cài đặt |
| `README.md` | Hướng dẫn cài đặt và sử dụng chương trình |

## 6. Link Video Demo

- **Video chạy thử chương trình**: *(Dán link Google Drive hoặc YouTube quay màn hình demo của bạn vào đây)*
