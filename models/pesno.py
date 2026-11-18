import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.layers import UNO_Block_Hybrid

class PeSNO(nn.Module):
    """
    Physics-encoded Spectral Neural Operator (PeSNO).
    Features:
    1. Hybrid Spectral Transforms (DST-I + FFT).
    2. Fourier Feature Embeddings for Grid.
    3. Hard Boundary Constraint Masks (Dirichlet).
    """
    def __init__(self, in_channels, out_channels, width, modes1, modes2):
        super().__init__()
        self.width = width
        
        # Fourier Feature Configuration
        self.num_freqs = 10
        self.freq_dim = 2 * self.num_freqs * 2 
        current_in_channels = in_channels + self.freq_dim
        
        # Embedding
        self.fc0 = nn.Linear(current_in_channels, width)
        self.norm0 = nn.InstanceNorm2d(width)
        
        # Encoder
        self.enc1 = UNO_Block_Hybrid(width, width, modes1, modes2)
        self.enc2 = UNO_Block_Hybrid(width, width*2, modes1, modes2)
        self.enc3 = UNO_Block_Hybrid(width*2, width*4, modes1, modes2)
        self.bottleneck = UNO_Block_Hybrid(width*4, width*4, modes1, modes2)
        
        # Decoder
        self.dec3 = UNO_Block_Hybrid(width*8, width*2, modes1, modes2)
        self.dec2 = UNO_Block_Hybrid(width*4, width, modes1, modes2)
        self.dec1 = UNO_Block_Hybrid(width*2, width, modes1, modes2)
        
        # Projection
        self.down1 = nn.Conv2d(width, width, 3, stride=2, padding=1)
        self.down2 = nn.Conv2d(width*2, width*2, 3, stride=2, padding=1)
        self.down3 = nn.Conv2d(width*4, width*4, 3, stride=2, padding=1)
        
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)
        self.output_scale = nn.Parameter(torch.tensor(1.0))

    def compute_fourier_features(self, grid):
        # grid: [B, 2, H, W], grid[:,0] is X, grid[:,1] is Y
        device = grid.device
        scales = torch.tensor([2**i for i in range(self.num_freqs)], device=device).float() * np.pi
        
        x_raw = grid[:, 0:1, :, :] # X (0~4)
        y_raw = grid[:, 1:2, :, :] # Y (0~1)
        x = x_raw / 4.0 # Normalize to 0~1
        y = y_raw
        
        out_list = []
        for s in scales:
            out_list.append(torch.sin(x * s)); out_list.append(torch.cos(x * s))
            out_list.append(torch.sin(y * s)); out_list.append(torch.cos(y * s))
            
        return torch.cat(out_list, dim=1)

    def forward(self, x, p_vec):
        # 1. Fourier Features
        grid_input = x[:, 1:3, :, :] 
        ff = self.compute_fourier_features(grid_input)
        x_aug = torch.cat([x, ff], dim=1) 
        
        # 2. Embedding
        x = x_aug.permute(0, 2, 3, 1)
        x = self.norm0(self.fc0(x).permute(0, 3, 1, 2))
        
        # 3. U-Net Path
        x1 = self.enc1(x, p_vec)
        x2 = self.enc2(self.down1(x1), p_vec)
        x3 = self.enc3(self.down2(x2), p_vec)
        xm = self.bottleneck(self.down3(x3), p_vec)
        
        def up(u, s, b): 
            u_up = F.interpolate(u, size=s.shape[-2:], mode='bilinear', align_corners=True)
            return b(torch.cat([u_up, s], 1), p_vec)
            
        x = up(xm, x3, self.dec3)
        x = up(x, x2, self.dec2)
        x = up(x, x1, self.dec1)
        
        # 4. Output
        x = x.permute(0, 2, 3, 1)
        x = self.fc2(F.gelu(self.fc1(x))).permute(0, 3, 1, 2)
        
        # Hard BC Mask (Dirichlet on Top/Bottom)
        if not hasattr(self, 'bc_mask') or self.bc_mask.shape[0] != x.shape[0]:
            self.bc_mask = torch.ones_like(x).detach()
            self.bc_mask[:, :, 0, :] = 0.0
            self.bc_mask[:, :, -1, :] = 0.0
            
        return x * self.bc_mask * self.output_scale