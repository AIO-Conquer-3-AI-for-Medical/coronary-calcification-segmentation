# model_def.py
import torch
import torch.nn as nn
import numpy as np

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.block(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class UNetModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=16):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base_channels)              
        self.enc2 = DownBlock(base_channels, base_channels * 2)         
        self.enc3 = DownBlock(base_channels * 2, base_channels * 4)     
        self.enc4 = DownBlock(base_channels * 4, base_channels * 8)     

        self.bottleneck = DownBlock(base_channels * 8, base_channels * 16) 

        self.up4 = UpBlock(in_channels=base_channels * 16, skip_channels=base_channels * 8, out_channels=base_channels * 8)
        self.up3 = UpBlock(in_channels=base_channels * 8, skip_channels=base_channels * 4, out_channels=base_channels * 4)
        self.up2 = UpBlock(in_channels=base_channels * 4, skip_channels=base_channels * 2, out_channels=base_channels * 2)
        self.up1 = UpBlock(in_channels=base_channels * 2, skip_channels=base_channels, out_channels=base_channels)

        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        b = self.bottleneck(s4)
        x = self.up4(b, s4)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        return self.out_conv(x)

    def predict_volume(self, images_25d_stack, hu_volume, device, threshold=0.5):
        """
        Thực hiện inference (dự đoán) và đóng gói cấu trúc đầu ra đồng bộ 100% 
        giữa mặt nạ phân đoạn và logic trích xuất tỷ trọng lâm sàng.
        
        Tham số:
        ----------
        images_25d_stack : np.ndarray
            Mảng khối đầu vào 2.5D đã được chuẩn hóa, shape: [D, 3, 512, 512]
        hu_volume : np.ndarray
            Khối mảng giá trị Hounsfield Unit thô gốc (chưa scale), shape: [D, 512, 512]
        device : str hoặc torch.device
            Thiết bị xử lý tính toán ('cuda' hoặc 'cpu')
        threshold : float
            Ngưỡng nhị phân hóa lâm sàng (nhận động từ thanh trượt score_threshold trên UI)
        """
        import pandas as pd
        
        self.eval()
        pred_mask_volume = []
        slice_records = []
        
        D, H, W = hu_volume.shape
        
        with torch.no_grad():
            for i in range(images_25d_stack.shape[0]):
                # 1. Lan truyền tiến (Forward pass) qua mạng U-Net
                slice_tensor = torch.from_numpy(images_25d_stack[i]).unsqueeze(0).float().to(device)
                logits = self(slice_tensor)
                
                # 2. Tính toán bản đồ xác suất qua hàm Sigmoid giống hệt file .ipynb
                probs = torch.sigmoid(logits).cpu().numpy()[0, 0]
                
                # 3. ĐỒNG BỘ NGƯỠNG: Tạo mặt nạ nhị phân duy nhất dựa trên tham số truyền vào
                binary_mask = (probs >= threshold).astype(np.uint8)
                pred_mask_volume.append(binary_mask)
                
                # 4. Trích xuất thông tin dựa trên CHÍNH mặt nạ nhị phân đã đồng bộ ngưỡng
                if np.any(binary_mask > 0):
                    hu_slice = hu_volume[i]
                    
                    # Tìm đậm độ lớn nhất (Max HU) TRONG VÙNG MÔ HÌNH DỰ ĐOÁN ĐƯỢC (đã áp threshold)
                    max_hu = np.max(hu_slice[binary_mask > 0])
                    
                    # Tiêu chuẩn lọc tỷ trọng lâm sàng tối thiểu
                    if max_hu >= 130.0:
                        slice_records.append({
                            "slice_idx": i,
                            "max_hu": round(float(max_hu), 1)
                        })
                        
        # Gộp kết quả thành mảng Numpy 3D dạng [D, 512, 512]
        pred_mask_volume_np = np.array(pred_mask_volume, dtype=np.uint8)
        
        # Tạo DataFrame đầu ra đảm bảo cấu trúc dữ liệu luôn nhất quán
        if slice_records:
            slice_records_df = pd.DataFrame(slice_records)
        else:
            slice_records_df = pd.DataFrame(columns=["slice_idx", "max_hu"])
            
        return pred_mask_volume_np, slice_records_df