# src/data_processing.py
import os
from pathlib import Path
import numpy as np
import pydicom

class COCATransformer:
    def __init__(self, window_center=400, window_width=1200):
        """
        Khởi tạo bộ chuyển đổi dữ liệu chuỗi ảnh DICOM sang khối ảnh giải phẫu 2.5D.
        
        Các tham số window_center và window_width mặc định được thiết lập tương ứng với 
        dải Hounsfield Unit thấp nhất là -200.0 và cao nhất là 1000.0 HU.
        """
        self.wc = window_center
        self.ww = window_width
        # Đồng bộ hóa chính xác dải chặn (clipping margins) từ file huấn luyện gốc
        self.hu_low = -200.0
        self.hu_high = 1000.0

    def load_and_sort_dicom_folder(self, folder_path):
        """
        Quét thư mục chứa chuỗi ảnh DICOM, sắp xếp theo tọa độ trục Z thực tế.
        Trả về danh sách đường dẫn file đã được sắp xếp và thông số pixel spacing.
        """
        folder = Path(folder_path)
        slices_meta = []
        pixel_spacing = None

        for file_path in folder.rglob("*"):
            if file_path.is_file():
                try:
                    # Đọc lướt qua header để lấy thông tin hình học giải phẫu trục Z
                    ds = pydicom.dcmread(str(file_path), stop_before_pixels=True)
                    if "ImagePositionPatient" in ds:
                        z_pos = float(ds.ImagePositionPatient[2])
                        slices_meta.append((z_pos, file_path))
                        if pixel_spacing is None and "PixelSpacing" in ds:
                            pixel_spacing = [float(x) for x in ds.PixelSpacing]
                except Exception:
                    continue

        if not slices_meta:
            return [], None

        # Sắp xếp chuỗi ảnh tăng dần dựa trên tọa độ hình học thực tế ImagePositionPatient[2]
        slices_meta.sort(key=lambda x: x[0])
        sorted_paths = [item[1] for item in slices_meta]
        
        return sorted_paths, pixel_spacing

    def prepare_25d_volume(self, sorted_paths):
        """
        Trích xuất giá trị Hounsfield Unit (HU), thực hiện chuẩn hóa dải tuyến tính 
        và xếp chồng các lát cắt thành các khối 2.5D (3 kênh đầu vào).
        
        Tuân thủ cấu trúc phân phối của dataclean.ipynb:
        - Kênh 0: Lát cắt hiện tại (Target slice 'k')
        - Kênh 1: Lát cắt phía trước (Context slice 'k-1')
        - Kênh 2: Lát cắt phía sau (Context slice 'k+1')
        """
        slices_hu = []

        # Đọc dữ liệu pixel thô và quy đổi sang đơn vị chuẩn HU định lượng y tế
        for path in sorted_paths:
            try:
                ds = pydicom.dcmread(str(path))
                intercept = float(ds.RescaleIntercept) if "RescaleIntercept" in ds else 0.0
                slope = float(ds.RescaleSlope) if "RescaleSlope" in ds else 1.0
                hu_array = ds.pixel_array.astype(np.float32) * slope + intercept
                slices_hu.append(hu_array)
            except Exception:
                continue

        hu_volume = np.array(slices_hu, dtype=np.float32)
        D, H, W = hu_volume.shape

        # Tiến hành ép dải chặn và chuẩn hóa Min-Max tuyến tính về đoạn [0.0, 1.0]
        images_normalized = np.clip(hu_volume, self.hu_low, self.hu_high)
        images_normalized = (images_normalized - self.hu_low) / (self.hu_high - self.hu_low)

        # Cấu trúc hóa các khối ảnh trượt 2.5D
        volume_25d = []
        for k in range(D):
            # Xử lý lặp biên giải phẫu an toàn (Replicate boundary padding) nếu chạm đỉnh/đáy khối ảnh
            idx_prev = max(0, k - 1)
            idx_next = min(D - 1, k + 1)

            # Khớp chính xác 100% thứ tự các trục kênh giải phẫu giống hệt khâu Huấn luyện
            stack = np.stack([
                images_normalized[k],         # Kênh 0: Lát cắt hiện tại cần phân đoạn (Target)
                images_normalized[idx_prev],  # Kênh 1: Ngữ cảnh lát cắt liền trước (Context)
                images_normalized[idx_next]   # Kênh 2: Ngữ cảnh lát cắt liền sau (Context)
            ], axis=0)
            
            volume_25d.append(stack)

        return volume_25d, hu_volume