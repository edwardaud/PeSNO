import torch
import torch.nn.functional as F

class GridConverter:
    """Helper for MAC Grid operations."""
    @staticmethod
    def collocated_to_staggered(u_col, v_col):
        u_pad = F.pad(u_col, (0, 1, 0, 0), mode='circular')
        u_stag = 0.5 * (u_col + u_pad[..., 1:])
        v_pad = F.pad(v_col, (0, 0, 1, 1), mode='replicate')
        v_stag = 0.5 * (v_pad[..., :-1, :] + v_pad[..., 1:, :])
        return u_stag, v_stag

    @staticmethod
    def staggered_to_collocated(u_stag, v_stag):
        u_pad = F.pad(u_stag, (1, 0, 0, 0), mode='circular')
        u_col = 0.5 * (u_pad[..., :-1] + u_stag)
        v_col = 0.5 * (v_stag[..., :-1, :] + v_stag[..., 1:, :])
        return u_col, v_col

class PressureSolverMAC:
    """
    [Core Physics] Conjugate Gradient Poisson Solver for MAC Grid.
    Supports JIT compilation for speed.
    """
    def __init__(self, dx, dy, device):
        self.dx2, self.dy2 = dx**2, dy**2
        self.beta_x, self.beta_y = 1.0/self.dx2, 1.0/self.dy2
        self.alpha = 2.0 * (self.beta_x + self.beta_y)
        self.device = device
        self.epsilon = 1e-10
        
        # Try JIT Compilation
        try:
            self.laplacian_step = torch.compile(self._laplacian_kernel, mode="reduce-overhead")
        except:
            print("⚠️ [JIT] Compilation failed, using eager mode.")
            self.laplacian_step = self._laplacian_kernel

    def _laplacian_kernel(self, p):
        p_left  = torch.roll(p, shifts=1, dims=3)
        p_right = torch.roll(p, shifts=-1, dims=3)
        p_pad_y = F.pad(p, (0,0,1,1), mode='replicate')
        p_down  = p_pad_y[:,:,:-2,:] 
        p_up    = p_pad_y[:,:,2:,:]  
        
        Ap = self.alpha * p - \
             (self.beta_x * (p_left + p_right) + \
              self.beta_y * (p_up + p_down))
        Ap = Ap + self.epsilon * p
        return Ap

    def solve(self, div, x0=None, max_iter=10000, tol=1e-8):
        b = -div
        b = b - torch.mean(b, dim=(1,2,3), keepdim=True)
        
        if x0 is None: p = torch.zeros_like(b)
        else: p = x0.clone(); p = p - torch.mean(p, dim=(1,2,3), keepdim=True)
            
        r = b - self.laplacian_step(p)
        d = r.clone()
        rs_old = torch.sum(r * r)
        init_res = rs_old.sqrt().item()
        
        target_res = max(tol, 1e-6 * init_res)
        if init_res < target_res: return p

        for i in range(max_iter):
            Ad = self.laplacian_step(d)
            dAd = torch.sum(d * Ad)
            
            if dAd.abs() < 1e-30: break
            
            alpha_cg = rs_old / dAd
            p = p + alpha_cg * d
            r = r - alpha_cg * Ad
            
            rs_new = torch.sum(r * r)
            err = rs_new.sqrt().item()
            
            if err < target_res: break
            
            beta_cg = rs_new / rs_old
            d = r + beta_cg * d
            rs_old = rs_new
            
        p = p - torch.mean(p, dim=(1,2,3), keepdim=True)
        return p

# Global Solver Cache to avoid re-init in workers
MAC_SOLVER = None
def get_solver(dx, dy, device):
    global MAC_SOLVER
    if MAC_SOLVER is None:
        MAC_SOLVER = PressureSolverMAC(dx, dy, device)
    return MAC_SOLVER

def project_velocity_mac(u_col, v_col, dx, dy, device, p_prev=None):
    """
    Standard Helmholtz-Hodge Decomposition projection.
    """
    solver = get_solver(dx, dy, device)
    
    # 1. Collocated -> Staggered
    u_stag, v_stag = GridConverter.collocated_to_staggered(u_col, v_col)
    
    # 2. Pre-Correction BC (No Penetration)
    v_stag[..., 0, :] = 0.0
    v_stag[..., -1, :] = 0.0
    
    # 3. Divergence
    u_stag_left = torch.roll(u_stag, shifts=1, dims=3) 
    div_x = (u_stag - u_stag_left) / dx
    div_y = (v_stag[..., 1:, :] - v_stag[..., :-1, :]) / dy
    div = div_x + div_y
    
    # 4. Pressure Solve
    p = solver.solve(div, x0=p_prev, max_iter=10000, tol=1e-8)
    
    # 5. Projection
    p_right = torch.roll(p, shifts=-1, dims=3)
    p_center = p
    
    u_stag_new = u_stag - (p_right - p_center) / dx
    v_stag_new = v_stag.clone()
    v_stag_new[..., 1:-1, :] = v_stag[..., 1:-1, :] - (p[..., 1:, :] - p[..., :-1, :]) / dy
    
    # 6. Post-Correction BC
    v_stag_new[..., 0, :] = 0.0
    v_stag_new[..., -1, :] = 0.0
    
    # 7. Staggered -> Collocated
    u_final, v_final = GridConverter.staggered_to_collocated(u_stag_new, v_stag_new)
    
    return u_final.double(), v_final.double(), p.double(), div.double()

class StreamFunctionSolver:
    """
    [Core Physics] Poisson Solver specifically for Stream Function.
    BCs: 
    - X-Axis: Periodic (Circular)
    - Y-Axis: Dirichlet (0 at walls) -> Implies impermeable boundaries.
    """
    def __init__(self, dx, dy, device):
        self.dx2, self.dy2 = dx**2, dy**2
        self.beta_x, self.beta_y = 1.0/self.dx2, 1.0/self.dy2
        self.alpha = 2.0 * (self.beta_x + self.beta_y)
        self.device = device
        self.epsilon = 1e-10
        
        try:
            self.laplacian_step = torch.compile(self._laplacian_kernel, mode="reduce-overhead")
        except:
            self.laplacian_step = self._laplacian_kernel

    def _laplacian_kernel(self, psi):
        # X: Periodic
        psi_left  = torch.roll(psi, shifts=1, dims=3)
        psi_right = torch.roll(psi, shifts=-1, dims=3)
        
        # Y: Dirichlet (Pad 0) 
        psi_pad_y = F.pad(psi, (0,0,1,1), mode='constant', value=0.0)
        psi_down  = psi_pad_y[:,:,:-2,:] 
        psi_up    = psi_pad_y[:,:,2:,:]  
        
        Ap = self.alpha * psi - \
             (self.beta_x * (psi_left + psi_right) + \
              self.beta_y * (psi_up + psi_down))
        return Ap + self.epsilon * psi

    def solve(self, omega, x0=None, max_iter=5000, tol=1e-8):
        b = omega
        
        if x0 is None: psi = torch.zeros_like(b)
        else: psi = x0.clone()
            
        r = b - self.laplacian_step(psi)
        d = r.clone()
        rs_old = torch.sum(r * r)
        init_res = rs_old.sqrt().item()
        
        target_res = tol 
        if init_res < target_res: return psi

        for i in range(max_iter):
            Ad = self.laplacian_step(d)
            dAd = torch.sum(d * Ad)
            
            if dAd.abs() < 1e-30: break
            
            alpha_cg = rs_old / dAd
            psi = psi + alpha_cg * d
            r = r - alpha_cg * Ad
            
            rs_new = torch.sum(r * r)
            err = rs_new.sqrt().item()
            
            if err < target_res: break
            
            beta_cg = rs_new / rs_old
            d = r + beta_cg * d
            rs_old = rs_new
            
        return psi

def compute_vorticity(u, v, dx, dy):
    """
    Compute Vorticity: omega = dv/dx - du/dy.
    BCs:
    - X: Periodic
    - Y: Free-Slip (Replicate Padding for normal derivative)
    """
    # X direction (Periodic)
    v_pad_x = F.pad(v, (1, 1, 0, 0), mode='circular')
    dv_dx = (v_pad_x[..., 2:] - v_pad_x[..., :-2]) / (2 * dx)
    
    # Y direction (Free Slip / Replicate)
    u_pad_y = F.pad(u, (0, 0, 1, 1), mode='replicate') 
    du_dy = (u_pad_y[..., 2:, :] - u_pad_y[..., :-2, :]) / (2 * dy)
    
    return dv_dx - du_dy