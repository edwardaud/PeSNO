import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FiLMBlock(nn.Module):
    def __init__(self, channels, param_dim=3):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(param_dim, 64), nn.GELU(), nn.Linear(64, channels * 2))
    def forward(self, x, p_vec):
        gamma, beta = self.mlp(p_vec).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        return x * (1 + gamma) + beta

class SpectralConv2d_Hybrid(nn.Module):
    """
    [PeSNO Core] Hybrid Spectral Conv
    Path A: DST (Y) + FFT (X) for global mixing.
    Path B: Local Conv with Anti-Symmetric Padding.
    """
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 
        self.modes2 = modes2
        
        self.scale = 1.0 / np.sqrt(in_channels * out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        
        self.local_conv = nn.Conv2d(in_channels, out_channels, 3, padding=0) 
        nn.init.xavier_normal_(self.local_conv.weight)

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # --- Path A: Spectral (Hybrid DST/FFT) ---
        x_odd = torch.cat([x, -torch.flip(x, dims=[2])], dim=2)
        x_ft = torch.fft.rfft2(x_odd, norm='ortho') 
        
        out_ft = torch.zeros(B, self.out_channels, x_ft.size(-2), x_ft.size(-1), dtype=torch.cfloat, device=x.device)
        m1, m2 = min(self.modes1, x_ft.size(2)), min(self.modes2, x_ft.size(3))
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2])
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2])
        
        x_spectral = torch.fft.irfft2(out_ft, s=(2*H, W), norm='ortho')[:, :, :H, :]
        
        # --- Path B: Local Conv (Anti-Symmetric Padding) ---
        x_top = -x[:, :, 0:1, :]
        x_bot = -x[:, :, -1:, :]
        x_pad_y = torch.cat([x_top, x, x_bot], dim=2)
        x_pad = F.pad(x_pad_y, (1, 1, 0, 0), mode='circular')
        x_local = self.local_conv(x_pad)
        
        return x_spectral + x_local

class UNO_Block_Hybrid(nn.Module):
    def __init__(self, in_width, out_width, modes1, modes2):
        super().__init__()
        self.conv = SpectralConv2d_Hybrid(in_width, out_width, modes1, modes2)
        self.w = nn.Conv2d(in_width, out_width, 1) 
        self.norm = nn.InstanceNorm2d(out_width)
        self.film = FiLMBlock(out_width)
        self.act = nn.GELU()
        self.dropout = nn.Dropout2d(p=0.05)
        nn.init.xavier_normal_(self.w.weight)

    def forward(self, x, p_vec):
        x_out = self.conv(x) + self.w(x)
        x_out = self.norm(x_out)
        x_out = self.film(x_out, p_vec)
        return self.dropout(self.act(x_out))