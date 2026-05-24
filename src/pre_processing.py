# src/pre_processing.py
import numpy as np
import zipfile
import shutil
from pathlib import Path

def extract_zip_dicom(uploaded_file, extract_to_dir):
    """
    Giải nén trực tiếp file zip được upload từ Streamlit vào thư mục tạm.
    Loại bỏ các file ẩn hoặc file hệ thống không hợp lệ.
    """
    extract_path = Path(extract_to_dir)
    
    # Làm sạch thư mục tạm nếu đã tồn tại trước đó để tránh lẫn dữ liệu ca bệnh cũ
    if extract_path.exists():
        shutil.rmtree(extract_path)
    extract_path.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            # Loại bỏ file rác hệ thống (như __MACOSX hoặc .DS_Store)
            if "__MACOSX" in file_info.filename or file_info.filename.endswith('.DS_Store') or file_info.is_dir():
                continue
            
            # Giải nén tệp tin cụ thể
            zip_ref.extract(file_info, extract_path)
            
    return extract_path

def dicom_to_hu(dcm):
    """
    Chuyển đổi giá trị pixel thô của máy quét sang đơn vị mật độ Hounsfield Unit (HU)
    dựa trên siêu dữ liệu Rescale Slope và Rescale Intercept của tệp DICOM.
    """
    img = dcm.pixel_array.astype(np.float32)
    if hasattr(dcm, "RescaleSlope") and hasattr(dcm, "RescaleIntercept"):
        img = img * dcm.RescaleSlope + dcm.RescaleIntercept
    return img

def apply_windowing(hu_img, window_center=40, window_width=400):
    """
    Áp dụng bộ lọc dải cửa sổ tương phản (Windowing) tối ưu cho mạch vành.
    Chuẩn hóa dải giá trị HU sau lọc về khoảng [0, 1].
    """
    min_val = window_center - window_width / 2
    max_val = window_center + window_width / 2
    windowed = np.clip(hu_img, min_val, max_val)
    return (windowed - min_val) / (max_val - min_val)