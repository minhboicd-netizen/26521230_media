"""
BÀI TẬP LỚN: XÂY DỰNG ỨNG DỤNG TƯƠNG TÁC VỚI CAMERA SỬ DỤNG MEDIAPIPE
Đề tài: Ứng dụng Bảng vẽ trên không (Vẽ tranh bằng cử chỉ ngón tay)

Quy trình xử lý của chương trình:
Camera -> MediaPipe (Lấy 21 điểm khớp bàn tay) -> Thuật toán kiểm tra ngón tay duỗi/gập
-> Xác định chế độ (Vẽ nét / Chọn công cụ / Xóa nhanh) -> Hiển thị bảng vẽ lên màn hình
"""

import os
import math
import time
import urllib.request
import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    raise ImportError(
        "Chưa cài đặt thư viện mediapipe! Hãy chạy lệnh: pip install -r requirements.txt"
    )

# Kiểm tra phiên bản MediaPipe: hỗ trợ cả bản mới (Tasks API >= 1.0 trên Python 3.12+)
# lẫn bản cũ (mp.solutions.hands trên Python 3.9 - 3.11)
DUNG_SOLUTIONS_CU = hasattr(mp, "solutions") and hasattr(mp.solutions, "hands")

# Cấu hình kích thước cửa sổ và thanh công cụ phía trên
CHIEU_RONG = 1280
CHIEU_CAO = 720
CAO_THANH_CONG_CU = 90

# Danh sách các nút trên thanh công cụ: (Tên hiển thị, Màu BGR, Loại nút)
DANH_SACH_CONG_CU = [
    ("XANH DUONG", (255, 120, 0), "MAU"),
    ("XANH LA", (60, 200, 60), "MAU"),
    ("DO", (50, 50, 240), "MAU"),
    ("VANG", (0, 215, 255), "MAU"),
    ("CUC TAY", (0, 0, 0), "TAY"),
    ("XOA HET", (90, 90, 90), "XOA_HET"),
]

# Các đoạn nối giữa 21 điểm khớp bàn tay để vẽ khung xương lên màn hình
CAC_DOAN_NOI_BAN_TAY = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Ngón cái
    (0, 5), (5, 6), (6, 7), (7, 8),        # Ngón trỏ
    (0, 9), (9, 10), (10, 11), (11, 12),   # Ngón giữa
    (0, 13), (13, 14), (14, 15), (15, 16), # Ngón áp út
    (0, 17), (17, 18), (18, 19), (19, 20), # Ngón út
    (5, 9), (9, 13), (13, 17)              # Lòng bàn tay
]

LINK_MODEL_BAN_TAY = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
TEN_FILE_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")


def chuan_bi_file_model():
    """Tải tự động file mô hình nhận diện bàn tay (khoảng 7.5 MB) nếu dùng MediaPipe bản mới."""
    if not os.path.exists(TEN_FILE_MODEL):
        print("Đang tải file mô hình bàn tay của MediaPipe (chỉ tải 1 lần đầu tiên)...")
        urllib.request.urlretrieve(LINK_MODEL_BAN_TAY, TEN_FILE_MODEL)
        print("Đã tải xong file hand_landmarker.task!")
    return TEN_FILE_MODEL


def ve_khung_xuong_ban_tay(khung_hinh, danh_sach_21_diem):
    """Vẽ 21 điểm khớp và các đường nối bàn tay lên khung hình camera."""
    cao, rong, _ = khung_hinh.shape
    toa_do_pixel = [
        (int(diem.x * rong), int(diem.y * cao)) for diem in danh_sach_21_diem
    ]

    for d1, d2 in CAC_DOAN_NOI_BAN_TAY:
        cv2.line(khung_hinh, toa_do_pixel[d1], toa_do_pixel[d2], (80, 220, 100), 2, cv2.LINE_AA)

    for x, y in toa_do_pixel:
        cv2.circle(khung_hinh, (x, y), 5, (40, 60, 240), -1, cv2.LINE_AA)
        cv2.circle(khung_hinh, (x, y), 5, (255, 255, 255), 1, cv2.LINE_AA)


def tinh_khoang_cach(diem_1, diem_2):
    """
    Tính khoảng cách giữa 2 điểm trên mặt phẳng theo công thức hình học:
    d = can_bac_hai((x1 - x2)^2 + (y1 - y2)^2)
    """
    return math.hypot(diem_1.x - diem_2.x, diem_1.y - diem_2.y)


def kiem_tra_5_ngon_tay(khop):
    """
    Thuật toán xác định trạng thái của 5 ngón tay từ danh sách 21 điểm khớp:
    Trả về danh sách 5 số [Ngón cái, Ngón trỏ, Ngón giữa, Ngón áp út, Ngón út]
    Quy ước: 1 là ngón tay đang duỗi mở, 0 là ngón tay đang gập lại.

    Cách kiểm tra:
    - Với 4 ngón dài (Trỏ, Giữa, Áp út, Út): So sánh khoảng cách từ cổ tay (điểm 0)
      đến đầu ngón tay (điểm 8, 12, 16, 20) với khoảng cách từ cổ tay đến khớp giữa (điểm 6, 10, 14, 18).
      Nếu đầu ngón tay xa cổ tay hơn khớp giữa thì ngón đó đang duỗi (1).
    - Với ngón cái (điểm 4): So sánh khoảng cách từ đầu ngón cái đến gốc ngón út (điểm 17)
      với khoảng cách từ khớp ngón cái (điểm 3) đến gốc ngón út.
    """
    co_tay = khop[0]
    trang_thai = []

    # 1. Kiểm tra ngón cái (điểm 4 so với điểm 3 lấy mốc ở gốc ngón út điểm 17)
    goc_ngon_ut = khop[17]
    kc_dau_ngon_cai = tinh_khoang_cach(khop[4], goc_ngon_ut)
    kc_khop_ngon_cai = tinh_khoang_cach(khop[3], goc_ngon_ut)
    if kc_dau_ngon_cai > kc_khop_ngon_cai * 1.08:
        trang_thai.append(1)
    else:
        trang_thai.append(0)

    # 2. Kiểm tra 4 ngón dài: Trỏ (8-6), Giữa (12-10), Áp út (16-14), Út (20-18)
    dau_ngon = [8, 12, 16, 20]
    khop_giua = [6, 10, 14, 18]

    for id_dau, id_giua in zip(dau_ngon, khop_giua):
        kc_dau = tinh_khoang_cach(co_tay, khop[id_dau])
        kc_giua = tinh_khoang_cach(co_tay, khop[id_giua])
        if kc_dau > kc_giua * 1.12:
            trang_thai.append(1)
        else:
            trang_thai.append(0)

    return trang_thai


def xac_dinh_che_do(trang_thai_ngon):
    """
    Thuật toán chuyển trạng thái ngón tay thành chế độ điều khiển bảng vẽ:
    - Chế độ VẼ ("VE"): Chỉ giơ ngón trỏ (ngón trỏ = 1, ngón giữa = 0).
    - Chế độ CHỌN CÔNG CỤ ("CHON"): Giơ cả ngón trỏ và ngón giữa (trỏ = 1, giữa = 1, áp út = 0, út = 0).
    - Chế độ XÓA NHANH ("XOA_NHANH"): Xòe cả 5 ngón tay (tổng 5 ngón = 5) và giữ yên để xóa bảng.
    - Chế độ NGHỈ ("NGHI"): Nắm tay hoặc các tư thế khác (ngừng vẽ để di chuyển tay không bị dính nét).
    """
    ngon_cai, ngon_tro, ngon_giua, ngon_ap_ut, ngon_ut = trang_thai_ngon
    tong_so_ngon = sum(trang_thai_ngon)

    if tong_so_ngon == 5:
        return "XOA_NHANH"
    elif ngon_tro == 1 and ngon_giua == 1 and ngon_ap_ut == 0 and ngon_ut == 0:
        return "CHON"
    elif ngon_tro == 1 and ngon_giua == 0 and ngon_ap_ut == 0 and ngon_ut == 0:
        return "VE"
    else:
        return "NGHI"


def ve_thanh_cong_cu(khung_hinh, vi_tri_dang_chon, che_do_hien_tai, co_but, fps):
    """Vẽ thanh chọn màu, cục tẩy ở phía trên và thanh trạng thái ở phía dưới."""
    cao, rong, _ = khung_hinh.shape

    # Nền thanh công cụ phía trên
    cv2.rectangle(khung_hinh, (0, 0), (rong, CAO_THANH_CONG_CU), (40, 45, 55), -1)

    so_nut = len(DANH_SACH_CONG_CU)
    chieu_rong_nut = (rong - 40) // so_nut

    for i, (ten, mau, loai) in enumerate(DANH_SACH_CONG_CU):
        x1 = 20 + i * chieu_rong_nut + 6
        y1 = 12
        x2 = 20 + (i + 1) * chieu_rong_nut - 6
        y2 = CAO_THANH_CONG_CU - 12

        mau_hien_thi = (230, 230, 230) if loai == "TAY" else mau
        cv2.rectangle(khung_hinh, (x1, y1), (x2, y2), mau_hien_thi, -1)

        if i == vi_tri_dang_chon:
            cv2.rectangle(khung_hinh, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (255, 255, 255), 3)
        else:
            cv2.rectangle(khung_hinh, (x1, y1), (x2, y2), (120, 120, 120), 1)

        mau_chu = (20, 20, 20) if loai == "TAY" or ten == "VANG" else (255, 255, 255)
        cv2.putText(khung_hinh, ten, (x1 + 14, y1 + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, mau_chu, 2, cv2.LINE_AA)

    # Thanh thông tin hướng dẫn ở dưới đáy màn hình
    cv2.rectangle(khung_hinh, (0, cao - 45), (rong, cao), (40, 45, 55), -1)
    ten_che_do = {
        "VE": "DANG VE (1 ngon tro)",
        "CHON": "DI CHUYEN / CHON MAU (2 ngon)",
        "XOA_NHANH": "GIU 5 NGON DE XOA BANG",
        "NGHI": "TAM NGUNG VE"
    }.get(che_do_hien_tai, "CHUA THAY TAY")

    thong_tin = (
        f"Che do: {ten_che_do}  |  Co but: {co_but}px (+/- de doi)  |  "
        f"[C] Xoa bang  [S] Luu anh  [Q] Thoat  |  FPS: {int(fps)}"
    )
    cv2.putText(khung_hinh, thong_tin, (20, cao - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)


def tron_anh_bang_ve(khung_hinh_camera, lop_muc_ve):
    """Ghép lớp mực vẽ (nền đen) đè lên hình ảnh camera để nét vẽ hiển thị rõ ràng."""
    anh_xam = cv2.cvtColor(lop_muc_ve, cv2.COLOR_BGR2GRAY)
    _, mat_na_nguoc = cv2.threshold(anh_xam, 10, 255, cv2.THRESH_BINARY_INV)
    mat_na_nguoc = cv2.cvtColor(mat_na_nguoc, cv2.COLOR_GRAY2BGR)

    khung_hinh_camera = cv2.bitwise_and(khung_hinh_camera, mat_na_nguoc)
    khung_hinh_ket_qua = cv2.bitwise_or(khung_hinh_camera, lop_muc_ve)
    return khung_hinh_ket_qua


def main():
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CHIEU_RONG)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CHIEU_CAO)

    if not camera.isOpened():
        print("Không mở được camera! Vui lòng kiểm tra lại kết nối Webcam.")
        return

    # Khởi tạo bộ nhận diện bàn tay tùy theo phiên bản MediaPipe trên máy
    bo_nhan_dien_cu = None
    bo_nhan_dien_moi = None

    if DUNG_SOLUTIONS_CU:
        bo_nhan_dien_cu = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
    else:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        duong_dan_model = chuan_bi_file_model()
        tuy_chon = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=duong_dan_model),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7
        )
        bo_nhan_dien_moi = mp_vision.HandLandmarker.create_from_options(tuy_chon)

    lop_muc_ve = np.zeros((CHIEU_CAO, CHIEU_RONG, 3), dtype=np.uint8)

    vi_tri_cong_cu = 0
    mau_but = DANH_SACH_CONG_CU[0][1]
    loai_cong_cu = "MAU"
    co_but = 8
    co_tay_xoa = 45

    diem_cu_x, diem_cu_y = 0, 0
    he_so_lam_muot = 0.65

    dem_xoa_nhanh = 0
    thoi_gian_truoc = time.time()
    moc_thoi_gian_ms = 0

    print("=== ỨNG DỤNG BẢNG VẼ TRÊN KHÔNG (MEDIAPIPE HANDS) ===")
    print("- Giơ 1 ngón trỏ: Vẽ nét lên màn hình")
    print("- Giơ 2 ngón (trỏ + giữa): Di chuyển con trỏ hoặc đưa lên thanh trên cùng để chọn màu / cục tẩy")
    print("- Xòe cả 5 ngón tay giữ khoảng 1 giây: Xóa sạch bảng vẽ")
    print("- Phím [+]/[-]: Tăng/giảm cỡ bút | [C]: Xóa bảng | [S]: Lưu hình vẽ | [Q]: Thoát")

    while True:
        doc_duoc, khung_hinh = camera.read()
        if not doc_duoc:
            print("Không đọc được hình ảnh từ camera.")
            break

        khung_hinh = cv2.resize(khung_hinh, (CHIEU_RONG, CHIEU_CAO))
        khung_hinh = cv2.flip(khung_hinh, 1)

        anh_rgb = cv2.cvtColor(khung_hinh, cv2.COLOR_BGR2RGB)
        danh_sach_21_diem = None

        # Lấy 21 điểm khớp bàn tay từ MediaPipe
        if DUNG_SOLUTIONS_CU:
            ket_qua = bo_nhan_dien_cu.process(anh_rgb)
            if ket_qua.multi_hand_landmarks:
                danh_sach_21_diem = ket_qua.multi_hand_landmarks[0].landmark
        else:
            moc_thoi_gian_ms = max(moc_thoi_gian_ms + 1, int(time.time() * 1000))
            anh_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=anh_rgb)
            ket_qua = bo_nhan_dien_moi.detect_for_video(anh_mp, moc_thoi_gian_ms)
            if ket_qua.hand_landmarks:
                danh_sach_21_diem = ket_qua.hand_landmarks[0]

        che_do = "CHUA_THAY_TAY"

        if danh_sach_21_diem is not None:
            ve_khung_xuong_ban_tay(khung_hinh, danh_sach_21_diem)

            x_tro = int(danh_sach_21_diem[8].x * CHIEU_RONG)
            y_tro = int(danh_sach_21_diem[8].y * CHIEU_CAO)
            x_giua = int(danh_sach_21_diem[12].x * CHIEU_RONG)
            y_giua = int(danh_sach_21_diem[12].y * CHIEU_CAO)

            trang_thai_ngon = kiem_tra_5_ngon_tay(danh_sach_21_diem)
            che_do = xac_dinh_che_do(trang_thai_ngon)

            # 1. Chế độ CHỌN CÔNG CỤ (Giơ 2 ngón: trỏ và giữa)
            if che_do == "CHON":
                diem_cu_x, diem_cu_y = 0, 0
                dem_xoa_nhanh = 0
                tb_x = (x_tro + x_giua) // 2
                tb_y = (y_tro + y_giua) // 2
                cv2.rectangle(khung_hinh, (tb_x - 18, tb_y - 18), (tb_x + 18, tb_y + 18), (255, 255, 255), 2)

                if tb_y < CAO_THANH_CONG_CU:
                    so_nut = len(DANH_SACH_CONG_CU)
                    chieu_rong_nut = (CHIEU_RONG - 40) // so_nut
                    chi_so = (tb_x - 20) // chieu_rong_nut
                    if 0 <= chi_so < so_nut:
                        ten, mau, loai = DANH_SACH_CONG_CU[chi_so]
                        if loai == "XOA_HET":
                            lop_muc_ve[:] = 0
                        else:
                            vi_tri_cong_cu = chi_so
                            mau_but = mau
                            loai_cong_cu = loai

            # 2. Chế độ VẼ (Chỉ giơ 1 ngón trỏ)
            elif che_do == "VE":
                dem_xoa_nhanh = 0
                if y_tro > CAO_THANH_CONG_CU:
                    if diem_cu_x == 0 and diem_cu_y == 0:
                        diem_cu_x, diem_cu_y = x_tro, y_tro

                    x_muot = int(he_so_lam_muot * x_tro + (1 - he_so_lam_muot) * diem_cu_x)
                    y_muot = int(he_so_lam_muot * y_tro + (1 - he_so_lam_muot) * diem_cu_y)

                    if loai_cong_cu == "TAY":
                        cv2.circle(khung_hinh, (x_muot, y_muot), co_tay_xoa // 2, (255, 255, 255), 2)
                        cv2.line(lop_muc_ve, (diem_cu_x, diem_cu_y), (x_muot, y_muot), (0, 0, 0), co_tay_xoa)
                    else:
                        cv2.circle(khung_hinh, (x_muot, y_muot), co_but, mau_but, -1)
                        cv2.line(lop_muc_ve, (diem_cu_x, diem_cu_y), (x_muot, y_muot), mau_but, co_but)

                    diem_cu_x, diem_cu_y = x_muot, y_muot
                else:
                    diem_cu_x, diem_cu_y = 0, 0

            # 3. Chế độ XÓA NHANH (Xòe cả 5 ngón tay giữ khoảng 20 khung hình)
            elif che_do == "XOA_NHANH":
                diem_cu_x, diem_cu_y = 0, 0
                dem_xoa_nhanh += 1
                cv2.putText(khung_hinh, f"Dang xoa bang: {min(100, dem_xoa_nhanh * 5)}%",
                            (CHIEU_RONG // 2 - 140, CHIEU_CAO // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 215, 255), 2, cv2.LINE_AA)
                if dem_xoa_nhanh >= 20:
                    lop_muc_ve[:] = 0
                    dem_xoa_nhanh = 0

            # 4. Chế độ NGHỈ
            else:
                diem_cu_x, diem_cu_y = 0, 0
                dem_xoa_nhanh = 0
        else:
            diem_cu_x, diem_cu_y = 0, 0
            dem_xoa_nhanh = 0

        khung_hinh = tron_anh_bang_ve(khung_hinh, lop_muc_ve)

        thoi_gian_nay = time.time()
        fps = 1.0 / max(1e-5, thoi_gian_nay - thoi_gian_truoc)
        thoi_gian_truoc = thoi_gian_nay

        ve_thanh_cong_cu(khung_hinh, vi_tri_cong_cu, che_do, co_but, fps)

        cv2.imshow("Bang Ve Tren Khong - MediaPipe", khung_hinh)

        phim = cv2.waitKey(1) & 0xFF
        if phim in (ord('q'), ord('Q'), 27):
            break
        elif phim in (ord('c'), ord('C')):
            lop_muc_ve[:] = 0
        elif phim in (ord('+'), ord('=')):
            co_but = min(30, co_but + 2)
        elif phim in (ord('-'), ord('_')):
            co_but = max(2, co_but - 2)
        elif phim in (ord('s'), ord('S')):
            ten_file = f"tranh_ve_{int(time.time())}.png"
            cv2.imwrite(ten_file, khung_hinh)
            print(f"Đã lưu ảnh màn hình vào file: {ten_file}")

    if bo_nhan_dien_cu is not None:
        bo_nhan_dien_cu.close()
    if bo_nhan_dien_moi is not None:
        bo_nhan_dien_moi.close()
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()