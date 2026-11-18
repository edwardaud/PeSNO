import torch
import torch.nn.functional as F
import numpy as np

def limiter_superbee(r):
    r = torch.nan_to_num(r, nan=0.0, posinf=2.0, neginf=0.0)
    val1 = torch.min(torch.ones_like(r), 2.0 * r)
    val2 = torch.min(2.0 * torch.ones_like(r), r)
    return torch.max(torch.zeros_like(r), torch.max(val1, val2))

def reconstruct_states(T, limiter_func, temp_idx_0, temp_idx_last):
    """
    High-order reconstruction for advection terms using Superbee limiter.
    """
    B, C, H, W = T.shape
    epsilon = 1e-12
    
    # X-Direction Reconstruction
    T_pad_x = F.pad(T, (2, 2, 0, 0), mode='circular') 
    T_left, T_center, T_right = T_pad_x[..., 1:-3], T, T_pad_x[..., 3:-1]
    delta_m = T_center - T_left; delta_p = T_right - T_center
    phi_L = limiter_func(delta_m / (delta_p + epsilon))
    T_L = T_center + 0.5 * phi_L * delta_p
    T_rr = T_pad_x[..., 4:]
    delta_pp = T_rr - T_right
    phi_R = limiter_func(delta_p / (delta_pp + epsilon))
    T_R = T_right - 0.5 * phi_R * delta_pp
    
    # Y-Direction Reconstruction (Dirichlet BCs)
    pad_0 = torch.full((B, C, 2, W), temp_idx_0, device=T.device, dtype=T.dtype)
    pad_last = torch.full((B, C, 2, W), temp_idx_last, device=T.device, dtype=T.dtype)
    T_pad_y = torch.cat([pad_0, T, pad_last], dim=2)
    T_up, T_cy, T_down = T_pad_y[..., 1:-3, :], T, T_pad_y[..., 3:-1, :]
    delta_u = T_cy - T_up; delta_d = T_down - T_cy
    phi_U = limiter_func(delta_u / (delta_d + epsilon))
    T_U = T_cy + 0.5 * phi_U * delta_d
    T_dd = T_pad_y[..., 4:, :]
    delta_dd = T_dd - T_down
    phi_D = limiter_func(delta_d / (delta_dd + epsilon))
    T_D = T_down - 0.5 * phi_D * delta_dd
    
    return T_L, T_R, T_U, T_D

def compute_rhs_hybrid(T, u, v, dx, dy, q_val, temp_idx_0, temp_idx_last):
    T, u, v = T.double(), u.double(), v.double()
    
    # Diffusion
    T_pad_lr = F.pad(T, (1, 1, 0, 0), mode='circular') 
    pad_0 = torch.full((T.shape[0], T.shape[1], 1, T.shape[3]+2), temp_idx_0, device=T.device, dtype=T.dtype)
    pad_last = torch.full((T.shape[0], T.shape[1], 1, T.shape[3]+2), temp_idx_last, device=T.device, dtype=T.dtype)
    T_pad = torch.cat([pad_0, T_pad_lr, pad_last], dim=2)
    
    d2x = (T_pad[:,:,1:-1, 2:] - 2*T + T_pad[:,:,1:-1, :-2]) / dx**2
    d2y = (T_pad[:,:,2:, 1:-1] - 2*T + T_pad[:,:,:-2, 1:-1]) / dy**2
    diffusion = d2x + d2y
    
    # Advection
    T_L, T_R, T_U, T_D = reconstruct_states(T, limiter_superbee, temp_idx_0, temp_idx_last)
    
    u_face = 0.5 * (F.pad(u, (0,1,0,0), mode='circular')[..., :-1] + F.pad(u, (0,1,0,0), mode='circular')[..., 1:])
    v_face = 0.5 * (F.pad(v, (0,0,0,1), mode='replicate')[..., :-1, :] + F.pad(v, (0,0,0,1), mode='replicate')[..., 1:, :])
    
    Flux_x = torch.relu(u_face)*T_L - torch.relu(-u_face)*T_R
    Flux_y = torch.relu(v_face)*T_U - torch.relu(-v_face)*T_D
    
    div_x = (Flux_x - F.pad(Flux_x, (1,0,0,0), mode='circular')[..., :-1]) / dx 
    div_y = (Flux_y - F.pad(Flux_y, (0,0,1,0), mode='replicate')[..., :-1, :]) / dy
    
    return diffusion - (div_x + div_y) + q_val

def rk4_step_adaptive(T, u, v, target_dt, dx, dy, q_val, temp_idx_0, temp_idx_last):
    T, u, v = T.double(), u.double(), v.double()
    max_vel = max(torch.max(torch.abs(u)).item(), torch.max(torch.abs(v)).item(), 1e-6)
    
    cfl_factor = 0.4 
    dt_safe = min(cfl_factor * min(dx, dy) / max_vel, cfl_factor * (dy ** 2) / 4.0)
    num_substeps = int(np.ceil(target_dt / dt_safe))
    if num_substeps > 2000: num_substeps = 2000
    real_dt = target_dt / num_substeps
    
    T_curr = T.clone()
    for _ in range(num_substeps):
        def rhs(t_in):
            return compute_rhs_hybrid(t_in, u, v, dx, dy, q_val, temp_idx_0, temp_idx_last)
            
        k1 = rhs(T_curr)
        k2 = rhs(T_curr + 0.5 * real_dt * k1)
        k3 = rhs(T_curr + 0.5 * real_dt * k2)
        k4 = rhs(T_curr + real_dt * k3)
        T_curr = T_curr + (real_dt/6.0)*(k1+2*k2+2*k3+k4)
        
        # Enforce Bounds & BCs
        T_curr = torch.clamp(T_curr, 0.0, 1.0)
        T_curr[:,:,0,:] = temp_idx_0
        T_curr[:,:,-1,:] = temp_idx_last
        
    return T_curr.double()