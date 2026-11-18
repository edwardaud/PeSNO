import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """
    [Baseline Component] Standard Residual Block with BN and ReLU
    Used to build the CNN encoder/decoder backbone.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        
        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        res = self.shortcut(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act(x + res)

class ResUNet(nn.Module):
    """
    [Baseline Model] Res U-Net (CNN based)
    Ablations relative to PeSNO:
    1. Local Convolutions (No Spectral/Global mixing).
    2. Naive Parameter Injection (Concatenation instead of FiLM).
    3. No Hard Boundary Constraints (Output is raw convolution).
    """
    def __init__(self, in_channels, out_channels, base_width=32):
        super().__init__()
        
        # Naive Fusion: Input channels + 3 Physics Parameters
        self.input_dim = in_channels + 3
        
        # Encoder
        self.in_conv = nn.Conv2d(self.input_dim, base_width, 3, padding=1)
        self.bn0 = nn.BatchNorm2d(base_width)
        self.act = nn.ReLU(inplace=True)
        
        self.enc1 = ResBlock(base_width, base_width)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = ResBlock(base_width, base_width*2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = ResBlock(base_width*2, base_width*4)
        self.pool3 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = ResBlock(base_width*4, base_width*8)
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(base_width*8, base_width*4, 2, stride=2)
        self.dec3 = ResBlock(base_width*8, base_width*4) 
        
        self.up2 = nn.ConvTranspose2d(base_width*4, base_width*2, 2, stride=2)
        self.dec2 = ResBlock(base_width*4, base_width*2) 
        
        self.up1 = nn.ConvTranspose2d(base_width*2, base_width, 2, stride=2)
        self.dec1 = ResBlock(base_width*2, base_width) 
        
        self.out_conv = nn.Conv2d(base_width, out_channels, 1)

    def forward(self, x, p_vec):
        """
        x: [B, 8, H, W]
        p_vec: [B, 3]
        """
        # --- 1. Naive Parameter Injection (Concat) ---
        B, _, H, W = x.shape
        p_map = p_vec.view(B, 3, 1, 1).expand(B, 3, H, W)
        x = torch.cat([x, p_map], dim=1) # [B, 11, H, W]
        
        # --- 2. Encoder ---
        x0 = self.act(self.bn0(self.in_conv(x)))
        
        x1 = self.enc1(x0)
        p1 = self.pool1(x1)
        
        x2 = self.enc2(p1)
        p2 = self.pool2(x2)
        
        x3 = self.enc3(p2)
        p3 = self.pool3(x3)
        
        # --- 3. Bottleneck ---
        bn = self.bottleneck(p3)
        
        # --- 4. Decoder with Adaptive Padding ---
        # Up 3
        u3 = self.up3(bn)
        # Handle Odd Dimensions (e.g. 506 -> 253 -> 126)
        diffY = x3.size(2) - u3.size(2)
        diffX = x3.size(3) - u3.size(3)
        u3 = F.pad(u3, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        d3 = self.dec3(torch.cat([x3, u3], dim=1))
        
        # Up 2
        u2 = self.up2(d3)
        diffY = x2.size(2) - u2.size(2)
        diffX = x2.size(3) - u2.size(3)
        u2 = F.pad(u2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        d2 = self.dec2(torch.cat([x2, u2], dim=1))
        
        # Up 1
        u1 = self.up1(d2)
        diffY = x1.size(2) - u1.size(2)
        diffX = x1.size(3) - u1.size(3)
        u1 = F.pad(u1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        d1 = self.dec1(torch.cat([x1, u1], dim=1))
        
        # --- 5. Output (No Hard BC) ---
        return self.out_conv(d1)