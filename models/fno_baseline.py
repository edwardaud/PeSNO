import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv2d_Vanilla(nn.Module):
    """
    [Baseline Component] Standard Spectral Convolution
    - Standard RFFT/IRFFT (Implicit Periodic BC)
    - No Hybrid Path (No local convolution branch)
    - No Anti-Symmetric Padding
    """
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Standard RFFT
        x_ft = torch.fft.rfft2(x, norm='ortho')
        
        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1, device=x.device, dtype=torch.cfloat)
        
        # Mode Selection & Complex Mul
        m1 = min(self.modes1, x_ft.size(2))
        m2 = min(self.modes2, x_ft.size(3))
        
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2])
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2])

        # IRFFT
        x = torch.fft.irfft2(out_ft, s=(H, W), norm='ortho')
        return x

class VanillaFNO2d(nn.Module):
    """
    [Baseline Model] Vanilla FNO
    Ablations relative to PeSNO:
    1. No FiLM (uses concatenation for parameters)
    2. No Hybrid Spectral Layers
    3. No Hard Boundary Constraints
    4. No Fourier Feature Embeddings for Grid
    """
    def __init__(self, in_channels, out_channels, width=32, modes1=16, modes2=32):
        super().__init__()
        
        # Naive Fusion: Input channels + 3 Physics Parameters
        self.input_dim = in_channels + 3 
        self.width = width
        
        # Lift
        self.fc0 = nn.Linear(self.input_dim, width)
        
        # Fourier Layers
        self.conv0 = SpectralConv2d_Vanilla(width, width, modes1, modes2)
        self.conv1 = SpectralConv2d_Vanilla(width, width, modes1, modes2)
        self.conv2 = SpectralConv2d_Vanilla(width, width, modes1, modes2)
        self.conv3 = SpectralConv2d_Vanilla(width, width, modes1, modes2)
        
        # Skip Connections
        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)

        # Projection
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x, p_vec):
        """
        x: [B, 8, H, W]
        p_vec: [B, 3]
        """
        # --- 1. Naive Parameter Injection (Concatenation) ---
        B, _, H, W = x.shape
        p_map = p_vec.view(B, 3, 1, 1).expand(B, 3, H, W)
        x = torch.cat([x, p_map], dim=1) # -> [B, 11, H, W]
        
        # --- 2. Lifting ---
        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        # --- 3. Fourier Iterations ---
        x1 = self.conv0(x) + self.w0(x)
        x = F.gelu(x1)

        x2 = self.conv1(x) + self.w1(x)
        x = F.gelu(x2)

        x3 = self.conv2(x) + self.w2(x)
        x = F.gelu(x3)

        x4 = self.conv3(x) + self.w3(x)
        x = F.gelu(x4)

        # --- 4. Projection ---
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)
        
        return x