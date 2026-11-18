import torch

def weighted_sobolev_loss(pred_psi_norm, target_psi_norm, validator, epoch):
    """
    [PeSNO Loss] Weighted H2 Sobolev Loss (Stream + Velocity + Vorticity)
    """
    with torch.no_grad():
        u_gt, v_gt = validator.compute_velocity_via_staggered(target_psi_norm, normalized=True)
        omega_gt = validator.compute_vorticity(u_gt, v_gt)
        
        # Spatial Weighting based on Velocity Magnitude (Plume focus)
        vel_mag = torch.sqrt(u_gt**2 + v_gt**2)
        weight_map = 1.0 + 2.0 * vel_mag 
        weight_map = weight_map / (weight_map.mean() + 1e-9)

    # 0th Order: Stream Function
    loss_psi = (torch.abs(pred_psi_norm - target_psi_norm) * weight_map).mean()
    
    # 1st Order: Velocity
    u_pred, v_pred = validator.compute_velocity_via_staggered(pred_psi_norm, normalized=True)
    loss_u = (torch.abs(u_pred - u_gt) * weight_map).mean()
    loss_v = (torch.abs(v_pred - v_gt) * weight_map).mean()
    
    # 2nd Order: Vorticity
    omega_pred = validator.compute_vorticity(u_pred, v_pred)
    loss_omega = (torch.abs(omega_pred - omega_gt) * weight_map).mean()
    
    # Curriculum Learning Schedule
    w_sob = min(5.0, 5.0 * (epoch / 20.0))
    w_omg = min(0.5, 0.5 * (max(0, epoch - 5) / 25.0)) 
    
    total_loss = loss_psi + w_sob * (loss_u + loss_v) + w_omg * loss_omega

    return total_loss, loss_psi.item(), (loss_u + loss_v).item(), loss_omega.item()

def evaluate_mantle_physics(model, loader, validator, device):
    model.eval()
    metrics = {"l2_psi": 0.0, "l2_vel": 0.0, "l2_omega": 0.0, "err_flux": 0.0, "linf_vel": 0.0}
    total = 0
    with torch.no_grad():
        for x, y_norm, p_vec, u_gt_phys, v_gt_phys in loader:
            x, y_norm, p_vec = x.to(device), y_norm.to(device), p_vec.to(device)
            T_field = x[:, 0:1, :, :] 
            pred_norm = model(x, p_vec)
            
            psi_p = pred_norm * validator.psi_scale
            psi_g = y_norm * validator.psi_scale
            
            metrics["l2_psi"] += (torch.norm(psi_p - psi_g, p=2, dim=(1,2,3)) / (torch.norm(psi_g, p=2, dim=(1,2,3)) + 1e-9)).sum().item()
            u_p, v_p = validator.compute_derived_velocity_phys(psi_p)
            u_t, v_t = validator.compute_derived_velocity_phys(psi_g)
            
            err_v = torch.sqrt((u_p - u_t)**2 + (v_p - v_t)**2)
            base_v = torch.sqrt(u_t**2 + v_t**2) + 1e-9
            metrics["l2_vel"] += (torch.norm(err_v, p=2, dim=(1,2,3)) / torch.norm(base_v, p=2, dim=(1,2,3))).sum().item()
            
            flux_p = (v_p * T_field).mean(dim=(1,2,3)); flux_t = (v_t * T_field).mean(dim=(1,2,3))
            metrics["err_flux"] += (torch.abs(flux_p - flux_t) / (torch.abs(flux_t) + 1e-9)).sum().item()
            
            max_p = torch.sqrt(u_p**2 + v_p**2).flatten(1).max(dim=1).values
            max_t = torch.sqrt(u_t**2 + v_t**2).flatten(1).max(dim=1).values
            metrics["linf_vel"] += (torch.abs(max_p - max_t) / (max_t + 1e-9)).sum().item()
            
            omega_p = validator.compute_vorticity(u_p, v_p)
            omega_t = validator.compute_vorticity(u_t, v_t)
            metrics["l2_omega"] += (torch.norm(omega_p - omega_t, p=2, dim=(1,2,3)) / (torch.norm(omega_t, p=2, dim=(1,2,3)) + 1e-9)).sum().item()
            total += x.size(0)
    return {k: v / total for k, v in metrics.items()}