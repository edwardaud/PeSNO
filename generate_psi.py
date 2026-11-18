import torch
import os
import time
import torch.multiprocessing as mp

# Import Shared Configs
from configs.config import DATA_ROOT, H_GRID, W_GRID, DX, DY

# Import Core Physics
from utils.fluid_core import StreamFunctionSolver, compute_vorticity

# ==========================================
# Worker Logic
# ==========================================
def process_sim_psi(args):
    sim_info, device_id = args
    sim_id, mode = sim_info[0], sim_info[1]
    
    device = torch.device(device_id)
    sim_dir = os.path.join(DATA_ROOT, mode, f"sim_{sim_id}")
    target_file = os.path.join(sim_dir, "e1_psi_data_select_snaps.pt")
    
    if os.path.exists(target_file):
        return f"⏭️  [Sim {sim_id}] Psi exists."
    
    try:
        u_path = os.path.join(sim_dir, "e1_uprev_data_select_snaps.pt")
        v_path = os.path.join(sim_dir, "e1_vprev_data_select_snaps.pt")
        if not os.path.exists(u_path): return f"⚠️  [Sim {sim_id}] Missing U/V."
            
        # Force Double Precision
        u_tensor = torch.load(u_path, map_location=device).double()
        v_tensor = torch.load(v_path, map_location=device).double()
    except Exception as e:
        return f"❌ [Sim {sim_id}] Error: {e}"
        
    solver = StreamFunctionSolver(DX, DY, device)
    psi_list = []
    psi_curr = None
    
    t0 = time.time()
    num_frames = u_tensor.shape[0]
    
    for t in range(num_frames):
        u_frame = u_tensor[t:t+1]
        v_frame = v_tensor[t:t+1]
        
        # 1. Compute Vorticity Source
        omega = compute_vorticity(u_frame, v_frame, DX, DY)
        
        # 2. Solve Poisson: Laplacian(Psi) = Omega
        # Using previous frame as guess (warm start) speeds up CG
        psi_curr = solver.solve(omega, x0=psi_curr, max_iter=2000, tol=1e-8)
        
        psi_list.append(psi_curr.cpu().double())
        
        if t % 200 == 0:
            # Optional: Print progress inside worker (can be noisy)
            pass
            
    torch.save(torch.cat(psi_list, dim=0), target_file)
    return f"✅ [Sim {sim_id}] Done ({time.time()-t0:.1f}s)"

# ==========================================
# Main Driver
# ==========================================
if __name__ == "__main__":
    # Hardware Config
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n🌊 Stream Function Generator (Post-Processing)")
    print(f"📂 Root: {DATA_ROOT}")
    
    try:
        sims = torch.load(os.path.join(DATA_ROOT, "sims.pt"), weights_only=False)
    except FileNotFoundError:
        print(f"❌ Error: sims.pt not found in {DATA_ROOT}. Run generate_data.py first.")
        exit(1)
    
    # Parallel Config
    NUM_WORKERS = 12 
    tasks = [(s, DEVICE_STR) for s in sims]
    
    print(f"🔥 Launching {NUM_WORKERS} workers for {len(tasks)} simulations...")
    
    ctx = mp.get_context('spawn')
    with ctx.Pool(NUM_WORKERS) as pool:
        for res in pool.imap_unordered(process_sim_psi, tasks):
            print(res)
            
    print("🎉 All Stream Functions generated.")