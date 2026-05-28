# src/agatston_score.py
import os
import re
import plistlib
import numpy as np
import pandas as pd
import cv2
from scipy.ndimage import label

def parse_tuple_string(s):
    """Trích xuất các giá trị số từ chuỗi định dạng tuple trong XML (ví dụ: '(12.3, 45.6, -78.9)')"""
    nums = re.findall(r"-?\d+\.?\d*", str(s))
    return tuple(float(x) for x in nums)

def xml_plist_to_mask_volume(xml_path, slices, hu_volume):
    """
    Dịch chuyển tọa độ đa giác từ file XML Plist sang mặt nạ nhị phân 3D (Ground Truth Mask).
    Đã sửa đổi: Áp dụng thuật toán Z-matching semantic từ Notebook để chống lệch tầng tuyệt đối.
    """
    gt_mask_volume = np.zeros_like(hu_volume, dtype=np.uint8)
    D, H, W = hu_volume.shape
    
    if not os.path.exists(xml_path):
        return gt_mask_volume

    # Trích xuất mảng tọa độ Z thực tế từ chuỗi DICOM đã sắp xếp
    z_values = np.array([float(s.ImagePositionPatient[2]) for s in slices], dtype=np.float32)

    with open(xml_path, 'rb') as fp:
        plist_data = plistlib.load(fp)
        
    images_meta = plist_data.get('Images', [])
    if not images_meta:
        return gt_mask_volume

    for img_item in images_meta:
        image_index = img_item.get('ImageIndex', -1)
        rois = img_item.get('ROIs', [])
        
        for roi in rois:
            # Thuật toán Z-matching ưu tiên trích xuất tọa độ thực tế
            roi_z = None
            center = roi.get("Center", None)
            if center is not None:
                vals = parse_tuple_string(center)
                if len(vals) == 3:
                    roi_z = vals[2]
            
            # Nếu không có Center, thử lấy từ điểm đầu tiên của chuỗi đa giác Point_px
            points = roi.get('Point_px', [])
            if roi_z is None and points:
                vals = parse_tuple_string(points[0])
                if len(vals) == 3:
                    roi_z = vals[2]

            # Thực hiện ánh xạ tầng dựa trên khoảng cách hình học Z nhỏ nhất
            if roi_z is not None:
                slice_idx = int(np.argmin(np.abs(z_values - roi_z)))
            else:
                # Fallback nếu không tìm thấy thuộc độ hình học trong ROI
                slice_idx = image_index if 0 <= image_index < D else -1
                
            if slice_idx == -1 or slice_idx >= D:
                continue

            # Vẽ đa giác ROI lên mặt nạ của slice tương ứng
            if points:
                polygon_coords = []
                for p_str in points:
                    coords = parse_tuple_string(p_str)
                    if len(coords) >= 2:
                        # Điểm ảnh thường lưu ở dạng (X, Y) hoặc (X, Y, Z)
                        polygon_coords.append([int(round(coords[0])), int(round(coords[1]))])
                
                if polygon_coords:
                    pts = np.array(polygon_coords, dtype=np.int32)
                    # Vẽ đè mặt nạ nhị phân (Value = 1)
                    cv2.fillPoly(gt_mask_volume[slice_idx], [pts], 1)

    return gt_mask_volume


def calculate_agatston_score(mask_volume, hu_volume, pixel_spacing):
    """
    Tính toán Điểm số Agatston lâm sàng chuẩn cho toàn bộ thể tích (Volume) ca bệnh.
    Hàm xử lý đồng bộ và độc lập cho cả kết quả dự đoán của AI và nhãn Ground Truth (GT).
    
    Tham số:
    ----------
    mask_volume : np.ndarray
        Mặt nạ phân đoạn nhị phân (AI hoặc GT), shape: [D, H, W]
    hu_volume : np.ndarray
        Khối mảng giá trị Hounsfield Unit thô gốc (Không dùng mảng normalized), shape: [D, H, W]
    pixel_spacing : list hoặc tuple
        Thông số khoảng cách pixel vật lý [spacing_x, spacing_y] lấy từ thuộc tính DICOM (đơn vị: mm)
    """
    total_patient_score = 0.0
    slice_records = []
    
    D, H, W = hu_volume.shape
    # Tính diện tích vật lý thực tế của một pixel đơn lẻ trên lát cắt (mm²)
    pixel_area_mm2 = float(pixel_spacing[0]) * float(pixel_spacing[1])
    
    for z in range(D):
        mask_slice = mask_volume[z]
        hu_slice = hu_volume[z]
        
        if not np.any(mask_slice > 0):
            continue
            
        # TIÊU CHUẨN LÂM SÀNG 1: Lọc ngưỡng đậm độ canxi (chỉ xét các pixel được phân đoạn có HU >= 130)
        valid_calcification_mask = (mask_slice > 0) & (hu_slice >= 130.0)
        
        if not np.any(valid_calcification_mask):
            continue
            
        # Phân tách các cụm tổn thương biệt lập (Connected Components) trên lát cắt 2D sử dụng 8-connectivity
        labeled_mask, num_features = label(valid_calcification_mask)
        slice_score = 0.0
        
        for cluster_idx in range(1, num_features + 1):
            cluster_pixels = (labeled_mask == cluster_idx)
            num_pixels = np.sum(cluster_pixels)
            
            # TIÊU CHUẨN LÂM SÀNG 2: Diện tích thực tế của cụm tổn thương phải >= 1 mm²
            cluster_area_mm2 = num_pixels * pixel_area_mm2
            if cluster_area_mm2 < 0.5:
                continue
                
            # Tìm đậm độ lớn nhất (Max HU) bên trong cụm để gán trọng số mật độ nguy cơ (Weight Factor)
            max_hu = np.max(hu_slice[cluster_pixels])
            
            if 130 <= max_hu < 200:
                weight = 1
            elif 200 <= max_hu < 300:
                weight = 2
            elif 300 <= max_hu < 400:
                weight = 3
            else:  # max_hu >= 400
                weight = 4
                
            # Điểm của cụm = Diện tích (mm²) * Trọng số đậm độ
            cluster_score = cluster_area_mm2 * weight
            slice_score += cluster_score
            
        if slice_score > 0:
            total_patient_score += slice_score
            slice_records.append({
                "slice_idx": z,
                "agatston_score": round(slice_score, 2),
                "max_hu": round(float(np.max(hu_slice[valid_calcification_mask])), 1) if num_features > 0 else 0.0
            })
            
    # Trả về tổng điểm tích lũy của bệnh nhân và DataFrame chi tiết từng lát cắt để hiển thị giao diện/QA validation
    return total_patient_score, pd.DataFrame(slice_records)


def calculate_agatston_for_volume(pred_mask_volume, hu_volume, pixel_spacing):
    """
    Tính Agatston score và phân loại nguy cơ tim mạch theo hướng dẫn lâm sàng.
    Dùng cho hiển thị trên giao diện Streamlit.
    
    Trả về:
    --------
    total_score : float
        Điểm Agatston tổng cộng
    risk_label : str
        Phân loại rủi ro (Minimal, Mild, Moderate, Extensive)
    risk_color : str
        Mã màu hex (#RRGGBB) tương ứng với mức rủi ro
    df_slices : pd.DataFrame
        Thông tin chi tiết từng lát cắt
    """
    total_score, df_slices = calculate_agatston_score(pred_mask_volume, hu_volume, pixel_spacing)
    
    # Phân loại rủi ro dựa trên hướng dẫn lâm sàng CAC
    if total_score == 0:
        risk_label = "No Coronary Calcification (Zero)"
        risk_color = "#00AA00"  # Xanh lá
    elif total_score < 10:
        risk_label = "Minimal (1-10)"
        risk_color = "#00DD00"  # Xanh lá nhạt
    elif total_score < 100:
        risk_label = "Mild (10-100)"
        risk_color = "#FFCC00"  # Vàng
    elif total_score < 400:
        risk_label = "Moderate (100-400)"
        risk_color = "#FF8800"  # Cam
    else:
        risk_label = "Extensive (≥400)"
        risk_color = "#DD0000"  # Đỏ
    
    return total_score, risk_label, risk_color, df_slices