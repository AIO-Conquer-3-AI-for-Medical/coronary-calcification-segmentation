# src/agatston_score.py
import numpy as np
import pandas as pd
from scipy.ndimage import label

def calculate_agatston_for_volume(pred_mask_volume, hu_volume, pixel_spacing):
    """
    Tính toán chỉ số Agatston trên toàn bộ khối ảnh (Volume) của bệnh nhân.
    
    Tham số:
    ----------
    pred_mask_volume : np.ndarray
        Mảng nhị phân kết quả dự đoán từ U-Net [D, 512, 512] (1: Vôi hóa, 0: Nền).
    hu_volume : np.ndarray
        Mảng ma trận ảnh CT gốc chưa chuẩn hóa ở đơn vị Hounsfield Unit [D, 512, 512].
    pixel_spacing : list hoặc tuple
        Quy cách kích thước hình học [spacing_x, spacing_y] trích xuất từ Header pydicom.
        
    Trả về:
    -------
    tuple: (total_patient_score, risk_class, risk_color, df_slices)
    """
    total_patient_score = 0.0
    slice_records = []
    
    # Tính toán diện tích thực tế của một điểm ảnh (đơn vị: mm^2)
    pixel_area_mm2 = pixel_spacing[0] * pixel_spacing[1]
    
    # Duyệt kiểm tra tuần tự qua từng lát cắt (Slice Z)
    for z in range(pred_mask_volume.shape[0]):
        pred_mask = pred_mask_volume[z]
        hu_slice = hu_volume[z]
        
        # Nếu lát cắt hiện tại AI không phát hiện bất kỳ pixel vôi hóa nào, bỏ qua để tăng tốc
        if np.sum(pred_mask) == 0:
            continue
            
        # Thuật toán Connected Component Labeling (loang ảnh) để phân biệt các cụm tổn thương biệt lập
        labeled_mask, num_features = label(pred_mask)
        slice_score = 0.0
        
        for i in range(1, num_features + 1):
            cluster_pixels = (labeled_mask == i)
            
            # 1. Đo lường diện tích cụm tổn thương (Area)
            num_pixels = np.sum(cluster_pixels)
            cluster_area_mm2 = num_pixels * pixel_area_mm2
            
            # Tiêu chuẩn lâm sàng: Chỉ tính điểm cho các cụm tổn thương có diện tích thực >= 1 mm^2
            if cluster_area_mm2 < 1.0:
                continue
                
            # 2. Xác định giá trị đậm độ đỉnh Hounsfield (Max HU Value) bên trong cụm
            max_hu = np.max(hu_slice[cluster_pixels])
            
            # 3. Phân nhóm áp trọng số điểm Agatston y khoa (Ngưỡng chặn dưới bắt buộc là 130 HU)
            if max_hu < 130:
                weight = 0
            elif 130 <= max_hu < 200:
                weight = 1
            elif 200 <= max_hu < 300:
                weight = 2
            elif 300 <= max_hu < 400:
                weight = 3
            else:  # max_hu >= 400
                weight = 4
                
            # Điểm của cụm = Diện tích (mm^2) * Trọng số độ đậm đặc
            cluster_score = cluster_area_mm2 * weight
            slice_score += cluster_score
            
        # Lưu nhận ký nếu lát cắt có đóng góp điểm số
        if slice_score > 0:
            total_patient_score += slice_score
            slice_records.append({
                "Slice Index": z,
                "Agatston Score": round(slice_score, 2)
            })
            
    # Phân loại mức độ rủi ro mạch vành dựa trên tổng điểm định lượng tích lũy
    if total_patient_score == 0:
        risk_class = "No Risk (No Calcium Detected)"
        risk_color = "#16a34a"  # Xanh lá
    elif 1 <= total_patient_score <= 10:
        risk_class = "Minimal Risk"
        risk_color = "#2563eb"  # Xanh dương
    elif 11 <= total_patient_score <= 100:
        risk_class = "Mild Risk"
        risk_color = "#ea580c"  # Cam
    elif 101 <= total_patient_score <= 400:
        risk_class = "Moderate Risk"
        risk_color = "#dc2626"  # Đỏ
    else:
        risk_class = "Severe Risk"
        risk_color = "#7f1d1d"  # Đỏ sẫm đục
        
    df_slices = pd.DataFrame(slice_records) if slice_records else pd.DataFrame(columns=["Slice Index", "Agatston Score"])
    
    return total_patient_score, risk_class, risk_color, df_slices