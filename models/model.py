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
        Khởi chạy vòng lặp Inference cho toàn bộ các lát cắt của ca bệnh.
        
        Tham số:
        ----------
        images_25d_stack : np.ndarray
            Mảng dữ liệu đầu vào đã chuẩn hóa và bọc khối 2.5D, shape: [D, 3, 512, 512]
        hu_volume : np.ndarray
            Mảng dữ liệu ảnh CT ở đơn vị HU gốc, shape: [D, 512, 512]
        """
        self.eval()
        pred_mask_volume = []
        
        with torch.no_grad():
            for i in range(images_25d_stack.shape[0]):
                # Trích xuất lát cắt 2.5D hiện tại và đẩy lên thiết bị tính toán (GPU/CPU)
                slice_tensor = torch.from_numpy(images_25d_stack[i]).unsqueeze(0).float().to(device) # [1, 3, 512, 512]
                
                # Forward qua mạng U-Net
                logits = self.forward(slice_tensor)
                probs = torch.sigmoid(logits).cpu().numpy()[0, 0] # Lấy ma trận xác suất 2D [512, 512]
                
                # Áp ngưỡng tạo mặt nạ phân vùng nhị phân
                binary_mask = (probs >= threshold).astype(np.uint8)
                pred_mask_volume.append(binary_mask)
                
        return np.array(pred_mask_volume), hu_volume