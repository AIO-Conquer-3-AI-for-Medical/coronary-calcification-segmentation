# src/data_processing.py
import os
from pathlib import Path
import numpy as np
import pydicom

class COCATransformer:
    def __init__(self, window_center=400, window_width=1200):
        self.wc = window_center
        self.ww = window_width
        # Đồng bộ chính xác dải chặn từ file huấn luyện gốc trong Notebook
        self.hu_low = -200.0
        self.hu_high = 1000.0

    def load_and_sort_dicom_folder(self, folder_path):
        """
        Quét toàn bộ thư mục DICOM, lọc các file hợp lệ và 
        sắp xếp nghiêm ngặt theo tọa độ Z giải phẫu tăng dần.
        """
        folder = Path(folder_path)
        slices_meta = []

        for file_path in folder.rglob("*"):
            if file_path.is_file():
                try:
                    ds = pydicom.dcmread(str(file_path))
                    if hasattr(ds, "pixel_array") and hasattr(ds, "ImagePositionPatient"):
                        slices_meta.append(ds)
                except Exception:
                    continue

        if not slices_meta:
            raise RuntimeError(f"Thư mục {folder_path} không chứa dữ liệu DICOM hợp lệ.")

        # SỬA LỖI LỆCH TRỤC Z: Sắp xếp lát cắt theo tọa độ hình học Z tăng dần
        slices_meta.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        
        # Lấy thông số kích thước pixel thực tế (phục vụ tính diện tích mm^2)
        pixel_spacing = slices_meta[0].PixelSpacing if hasattr(slices_meta[0], "PixelSpacing") else [0.5, 0.5]
        
        return slices_meta, pixel_spacing

    def transform_patient_volume(self, sorted_slices):
        """
        Chuyển đổi chuỗi ảnh DICOM đã sắp xếp thành mảng HU thô và khối 2.5D bọc kênh.
        """
        slices_hu = []
        for ds in sorted_slices:
            intercept = float(ds.RescaleIntercept) if "RescaleIntercept" in ds else 0.0
            slope = float(ds.RescaleSlope) if "RescaleSlope" in ds else 1.0
            hu_array = ds.pixel_array.astype(np.float32) * slope + intercept
            slices_hu.append(hu_array)

        hu_volume = np.array(slices_hu, dtype=np.float32)
        D, H, W = hu_volume.shape

        # Tiến hành ép dải chặn (clipping) và chuẩn hóa tuyến tính giống Notebook huấn luyện
        images_normalized = np.clip(hu_volume, self.hu_low, self.hu_high)
        images_normalized = (images_normalized - self.hu_low) / (self.hu_high - self.hu_low)

        # SỬA LỖI KÊNH 2.5D: Sắp xếp chuẩn khít [Kênh 0: Trước, Kênh 1: Hiện tại, Kênh 2: Sau]
        volume_25d = []
        for k in range(D):
            idx_prev = max(0, k - 1)
            idx_next = min(D - 1, k + 1)
            
            # Cấu trúc mảng 3 kênh shape (3, H, W)
            slice_25d = np.stack([
                images_normalized[idx_prev],  # Kênh 0: t-1
                images_normalized[k],         # Kênh 1: t (Lát cắt đích)
                images_normalized[idx_next]   # Kênh 2: t+1
            ], axis=0)
            
            volume_25d.append(slice_25d)

        return np.array(volume_25d, dtype=np.float32), hu_volume