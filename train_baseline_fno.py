import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ==========================================
# 1. Import Shared Components
# ==========================================
from configs.config import setup_seed, DEVICE, DATA_ROOT
from configs.config import BATCH_SIZE, EPOCHS, START_LR, MIN_LR, WEIGHT_DECAY, PLOT_EVERY_EPOCH

from data.dataset import AugmentedStreamDataset
from utils.physics import ExactStreamPhysicsValidator
from utils.losses import weighted_sobolev_loss, evaluate_mantle_physics
from utils.visualizer import plot_stream_prediction

# Import Baseline Specific Model
from models.fno_baseline import VanillaFNO2d

# ==========================================
# 2. Baseline Specific Configs
# ==========================================
LOG_FILE = "training_log_baseline_FNO.txt"
VIS_DIR = "visuals_baseline_FNO"

# Architecture Params
WIDTH = 32
MODES1 = 16  # Y
MODES2 = 32  # X
IN_CHANNELS = 8 
OUT_CHANNELS = 1

# ==========================================
# 3. Main Loop
# ==========================================
if __name__ == "__main__":
    setup_seed(2025)
    os.makedirs(VIS_DIR, exist_ok=True)
    
    print("\n" + "="*90)
    print("📉  System: Vanilla FNO Baseline (No FiLM, No HardBC, No Hybrid)")
    print(f"📁  Log: {LOG_FILE}")
    print(f"🖼️  Vis: {VIS_DIR}")
    print("="*90)
    
    # --- Data Loading (Shared Logic) ---
    # Use probe=True for train to calculate stats
    train_ds = AugmentedStreamDataset(DATA_ROOT, 'train', probe=True)
    
    # IMPORTANT: Pass train stats to CV to ensure EXACT same normalization as PeSNO experiment
    stats = {'psi_norm': train_ds.psi_norm, 'dtdx_norm': train_ds.dtdx_norm}
    cv_ds = AugmentedStreamDataset(DATA_ROOT, 'cv', probe=False, external_stats=stats)
    
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    cv_loader = DataLoader(cv_ds, BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
    
    # --- Validator (Shared Logic) ---
    validator = ExactStreamPhysicsValidator(DEVICE, psi_scale=train_ds.psi_norm)
    print(f"🔄 Validator synced: PSI_NORM={train_ds.psi_norm:.4f}")
    
    # --- Model (Baseline Specific) ---
    print(f"🏗️  Initializing VanillaFNO2d...")
    model = VanillaFNO2d(IN_CHANNELS, OUT_CHANNELS, WIDTH, MODES1, MODES2).to(DEVICE)
    
    # --- Optimization ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=START_LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=MIN_LR)
    
    # --- Logging ---
    header = "{:<5} | {:<9} | {:<9} | {:<9} | {:<9} || {:<9} | {:<9} | {:<9} | {:<9} | {:<9}"
    print("-" * 125)
    print(header.format("Epoch", "Tr_Loss", "Tr_Psi", "Tr_Vel", "Tr_Omg", "Val_Psi", "Val_Vel", "Val_Omg", "V_Flux", "V_Linf"))
    print("-" * 125)
    
    with open(LOG_FILE, "w") as f:
        f.write("Epoch,Tr_Loss,Tr_Psi,Tr_Vel,Tr_Omg,Val_Psi,Val_Vel,Val_Omg,Val_Flux,Val_Linf\n")
    
    best_l2_vel = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        acc_loss, acc_psi, acc_vel, acc_omg = 0, 0, 0, 0
        
        for x, y_norm, p_vec, _, _ in train_loader:
            x, y_norm, p_vec = x.to(DEVICE), y_norm.to(DEVICE), p_vec.to(DEVICE)
            
            optimizer.zero_grad()
            # Forward pass (Baseline handles p_vec internally via concat)
            pred = model(x, p_vec)
            
            loss, l_p, l_v, l_o = weighted_sobolev_loss(pred, y_norm, validator, epoch)
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            acc_loss += loss.item(); acc_psi += l_p; acc_vel += l_v; acc_omg += l_o
            
        scheduler.step()
        
        # Validation
        metrics = evaluate_mantle_physics(model, cv_loader, validator, DEVICE)
        n = len(train_loader)
        
        # Log
        print(header.format(
            f"E{epoch:03d}", 
            f"{acc_loss/n:.4f}", f"{acc_psi/n:.4f}", f"{acc_vel/n:.4f}", f"{acc_omg/n:.4f}", 
            f"{metrics['l2_psi']:.4f}", f"{metrics['l2_vel']:.4f}", f"{metrics['l2_omega']:.4f}", 
            f"{metrics['err_flux']:.4f}", f"{metrics['linf_vel']:.4f}" 
        ))
        
        with open(LOG_FILE, "a") as f:
            f.write(f"{epoch},{acc_loss/n},{acc_psi/n},{acc_vel/n},{acc_omg/n},"
                    f"{metrics['l2_psi']},{metrics['l2_vel']},{metrics['l2_omega']},"
                    f"{metrics['err_flux']},{metrics['linf_vel']}\n")
            
        # Save Best
        if metrics['l2_vel'] < best_l2_vel:
            best_l2_vel = metrics['l2_vel']
            torch.save(model.state_dict(), os.path.join(VIS_DIR, "best_baseline_fno.pth"))
            print(f"  💾 Best Baseline L2: {best_l2_vel:.4f}")
            
        # Plotting (Using shared visualizer with Baseline Title injection if possible, 
        # or standard behavior)
        if epoch % PLOT_EVERY_EPOCH == 0:
            plot_stream_prediction(model, cv_ds, epoch, VIS_DIR, validator, DEVICE)
            
        torch.save(model.state_dict(), os.path.join(VIS_DIR, "latest_baseline.pth"))

    print("✅ Baseline Training Finished.")