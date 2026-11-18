import torch
import numpy as np
import random
import os

# ==========================================
# Global Hardware & Seed Setup
# ==========================================
def setup_seed(seed=2025):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True 
    print(f"✅ [Setup] Random Seed fixed to: {seed} (Deterministic Mode)")

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# Paths & Hyperparameters
# ==========================================
DATA_ROOT = r"/gz-data/TPH_Synthetic-Ultra-Precision-4"
LOG_FILE = "training_log_pesno.txt"
VIS_DIR = "visuals_pesno"

BATCH_SIZE = 32
EPOCHS = 100
START_LR = 1e-3
MIN_LR = 1e-6
WEIGHT_DECAY = 1e-4
PLOT_EVERY_EPOCH = 1

# ==========================================
# Physical Constants (Mantle Convection)
# ==========================================
H_GRID, W_GRID = 128, 506
DX = 4.0 / W_GRID
DY = 1.0 / H_GRID
VISCOSITY_SCALE_FACTOR = 20.0