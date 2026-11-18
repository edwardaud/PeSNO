import torch
import torch.nn.functional as F
from configs.config import DX, DY

class ExactStreamPhysicsValidator:
    """
    [Physics Core] Staggered Grid Finite Difference Scheme.
    Handles Psi -> Velocity conversion ensuring mass conservation.
    """
    def __init__(self, device, psi_scale=1.0):
        self.dx = DX
        self.dy = DY
        self.device = device
        self.psi_scale = psi_scale 
        
        # Pre-compute Laplacian coefficients if needed
        self.dx2, self.dy2 = self.dx**2, self.dy**2

    def compute_velocity_via_staggered(self, psi, normalized=False):
        """
        Psi (Center) -> U_stag (Face) -> U_col (Center)
        normalized=True: Returns grid-level velocity (for Loss calculation)
        normalized=False: Returns physical velocity (for Evaluation)
        """
        h_step = 1.0 if normalized else self.dy
        w_step = 1.0 if normalized else self.dx
        
        # 1. U = dPsi/dy
        psi_pad_y = F.pad(psi, (0,0,1,1), mode='constant', value=0.0)
        u_stag = (psi_pad_y[..., 1:, :] - psi_pad_y[..., :-1, :]) / h_step
        u_col = 0.5 * (u_stag[..., :-1, :] + u_stag[..., 1:, :])
        
        # 2. V = -dPsi/dx (Circular BC)
        psi_left = psi
        psi_right = torch.roll(psi, shifts=-1, dims=3)
        v_stag = -(psi_right - psi_left) / w_step
        
        v_stag_prev = torch.roll(v_stag, shifts=1, dims=3)
        v_col = 0.5 * (v_stag + v_stag_prev)
        
        return u_col, v_col

    def compute_vorticity(self, u, v):
        # Vorticity = dv/dx - du/dy
        v_pad_x = F.pad(v, (1,1,0,0), mode='circular')
        dv_dx = (v_pad_x[..., 2:] - v_pad_x[..., :-2]) / (2 * self.dx)
        
        u_pad_y = F.pad(u, (0,0,1,1), mode='replicate')
        du_dy = (u_pad_y[..., 2:, :] - u_pad_y[..., :-2, :]) / (2 * self.dy)
        return dv_dx - du_dy
    
    def compute_derived_velocity_phys(self, psi_phys):
        return self.compute_velocity_via_staggered(psi_phys, normalized=False)