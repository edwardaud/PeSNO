import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import random

def plot_stream_prediction(model, dataset, epoch, save_dir, validator, device):
    model.eval()
    idx = random.randint(0, len(dataset)-1)
    if hasattr(dataset, 'dataset'): # Handle Subset
        real_idx = dataset.indices[idx]; ds = dataset.dataset
    else: 
        real_idx = idx; ds = dataset
    
    x, y_norm, p_vec, u_gt_phys, v_gt_phys = ds[real_idx]
    
    with torch.no_grad():
        pred_norm = model(x.unsqueeze(0).to(device), p_vec.unsqueeze(0).to(device))
    
    psi_p = (pred_norm * validator.psi_scale).cpu().squeeze().numpy()
    psi_g = (y_norm * validator.psi_scale).squeeze().numpy()
    
    psi_phys_tensor = pred_norm * validator.psi_scale
    u_p, v_p = validator.compute_derived_velocity_phys(psi_phys_tensor)
    
    u_p = u_p.cpu().squeeze().numpy(); v_p = v_p.cpu().squeeze().numpy()
    u_g = u_gt_phys.squeeze().numpy(); v_g = v_gt_phys.squeeze().numpy()
    
    plot_kwargs = {'origin': 'lower', 'aspect': 'auto', 'interpolation': 'nearest'}
    fig, axs = plt.subplots(3, 3, figsize=(18, 12), dpi=120)
    
    def draw_row(row_idx, data_gt, data_pred, label, cmap):
        val_max = max(np.abs(data_gt).max(), np.abs(data_pred).max())
        val_min = -val_max
        im1 = axs[row_idx, 0].imshow(data_gt, cmap=cmap, vmin=val_min, vmax=val_max, **plot_kwargs)
        axs[row_idx, 0].set_title(f"GT {label}"); plt.colorbar(im1, ax=axs[row_idx, 0], fraction=0.046, pad=0.04)
        im2 = axs[row_idx, 1].imshow(data_pred, cmap=cmap, vmin=val_min, vmax=val_max, **plot_kwargs)
        axs[row_idx, 1].set_title(f"Pred {label}"); plt.colorbar(im2, ax=axs[row_idx, 1], fraction=0.046, pad=0.04)
        err = np.abs(data_gt - data_pred)
        im3 = axs[row_idx, 2].imshow(err, cmap='inferno', **plot_kwargs)
        axs[row_idx, 2].set_title(f"Abs Error (Max: {err.max():.2e})"); plt.colorbar(im3, ax=axs[row_idx, 2], fraction=0.046, pad=0.04)
        for col in range(3): axs[row_idx, col].set_xticks([]); axs[row_idx, col].set_yticks([])

    draw_row(0, psi_g, psi_p, "Psi", 'RdBu_r')
    draw_row(1, u_g, u_p, "U (Horiz)", 'PuOr')
    draw_row(2, v_g, v_p, "V (Vert)", 'PuOr')

    plt.suptitle(f"Epoch {epoch:03d} | Sim {real_idx} | PeSNO Eval", fontsize=12, y=1.0)
    plt.tight_layout(); plt.subplots_adjust(top=0.92)
    plt.savefig(os.path.join(save_dir, f"E{epoch:03d}_Physics_Viz.png")); plt.close()