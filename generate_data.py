import torch
import numpy as np
import os
import time
import torch.multiprocessing as mp
import sys

# ==========================================
# 0. 路径注入
# ==========================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 推导 teacher 文件夹路径
TEACHER_DIR = os.path.join(SCRIPT_DIR, "teacher")


if TEACHER_DIR not in sys.path:
    sys.path.append(TEACHER_DIR)

if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

# ==========================================
# 1. 统一配置入口
# ==========================================
from configs.config import DATA_ROOT, H_GRID, W_GRID, DX, DY
from utils.fluid_core import project_velocity_mac
from utils.numerical import rk4_step_adaptive

# 外部网络依赖
try:
    from pytorch_networks_convae import NewFluidNet, TS
except ImportError:
    raise ImportError(f"❌ Error: Could not find 'pytorch_networks_convae.py' in {TEACHER_DIR}")

# ==========================================
# 2. 生成任务专用配置
# ==========================================

RAW_DATA_ROOT = os.path.join(TEACHER_DIR, "TPH")
AUTHOR_WEIGHT_PATH = os.path.join(TEACHER_DIR, "81_fluidnet_uvp.pt")

STEPS_PER_SIM = 1000
SAVE_INTERVAL = 1
DT_PHYS = 1e-4

# 物理稳定性参数
INERTIA_ALPHA = 0.6         
SCALER_CAP = 5000.0            
TEMP_AT_IDX_0 = 1.0         
TEMP_AT_IDX_LAST = 0.0 

# Teacher Network Hyperparameters
NETWORK_PARAMS = {
    'levels': 5, 'repeats': 6, 'kernel': 5, 'c_h': 16, 'act_fn': "gelu",
    'r_p': "learned", 'loss_type': "curl", 'use_symm': False, 'dilation': 1,
    'a_bound': 10, 'use_skip': False, 'p_pred': False, 'blurr': False,
    'factor': 2, 'scale': True, 'net': "newfluidnet"
}

# ==========================================
# 3. 辅助函数
# ==========================================
def load_teacher_model(device):
    """Load the pre-trained FluidNet teacher model."""
    c_i = 7; c_o = 3
    if NETWORK_PARAMS['loss_type'] == "curl": c_o -= 1
    if not NETWORK_PARAMS['p_pred']: c_o -= 1

    model = NewFluidNet(
        levels=NETWORK_PARAMS['levels'], c_i=c_i, c_h=NETWORK_PARAMS['c_h'], c_o=c_o, 
        device=device, act_fn=NETWORK_PARAMS['act_fn'], r_p=NETWORK_PARAMS['r_p'], 
        loss_type=NETWORK_PARAMS['loss_type'], use_symm=NETWORK_PARAMS['use_symm'], 
        dilation=NETWORK_PARAMS['dilation'], a_bound=NETWORK_PARAMS['a_bound'], 
        repeats=NETWORK_PARAMS['repeats'], use_skip=NETWORK_PARAMS['use_skip'], 
        f=NETWORK_PARAMS['kernel'], p_pred=NETWORK_PARAMS['p_pred'], 
        blurr=NETWORK_PARAMS['blurr'], factor=NETWORK_PARAMS['factor']
    ).double().to(device)
    
    ts_net = TS(model, ad=None, device=device, ts=1, scale=True, p_pred=False, net="newfluidnet").double().to(device)
    
    # Robust loading
    if os.path.exists(AUTHOR_WEIGHT_PATH):
        try:
            ts_net.load_state_dict(torch.load(AUTHOR_WEIGHT_PATH, map_location=device))
            print(f"✅ Loaded weights from: {AUTHOR_WEIGHT_PATH}")
        except:
            model.load_state_dict(torch.load(AUTHOR_WEIGHT_PATH, map_location=device))
    else:
        raise FileNotFoundError(f"❌ Teacher weights not found at: {AUTHOR_WEIGHT_PATH}")
        
    ts_net.eval()
    return ts_net

def prepare_inputs(T, sim_info, device):
    """Normalize inputs for the Teacher network."""
    raq, fkt, fkp = sim_info[2], sim_info[3], sim_info[4]
    
    y = torch.linspace(0, 1, H_GRID, device=device).view(1, 1, H_GRID, 1).expand(1, 1, H_GRID, W_GRID).double()
    x = torch.linspace(0, 4, W_GRID, device=device).view(1, 1, 1, W_GRID).expand(1, 1, H_GRID, W_GRID).double()
    
    sdf = torch.zeros_like(y); sdf[:,:,0,:]=1.; sdf[:,:,-1,:]=1.; sdf[:,:,:,0]=1.; sdf[:,:,:,-1]=1.
    sdf2 = torch.ones_like(y); sdf2[:,:,0,:]=0.; sdf2[:,:,-1,:]=0.; sdf2[:,:,:,0]=0.; sdf2[:,:,:,-1]=0.
    
    raq_nd = torch.tensor((raq - 0.12624371) / (9.70723344 - 0.12624371), dtype=torch.float64, device=device)
    fkt_log = np.log10(fkt); fkp_log = np.log10(fkp)
    fkt_nd = torch.tensor((fkt_log - 6.00352841978384) / (9.888820429862925 - 6.00352841978384), dtype=torch.float64, device=device)
    fkp_nd = torch.tensor((fkp_log - 0.005251646002323797) / (1.9927988938926755 - 0.005251646002323797), dtype=torch.float64, device=device)
    
    return [
        T.double(), sdf, sdf2, y, raq_nd, fkt_nd, fkp_nd, raq, 
        torch.tensor(fkt, device=device, dtype=torch.float64), 
        torch.tensor(fkp, device=device, dtype=torch.float64), 
        x, y
    ]

# ==========================================
# 4. Worker Function
# ==========================================
def process_simulation(args):
    sim_idx, sim_info, device_id = args
    device = torch.device(device_id)
    sim_id, mode = sim_info[0], sim_info[1]
    q_val = sim_info[2]
    
    target_dir = os.path.join(DATA_ROOT, mode, f"sim_{sim_id}")
    if os.path.exists(target_dir): return f"⏭️  Skipped Sim {sim_id} (Already exists)"

    try:
        teacher = load_teacher_model(device)
    except Exception as e:
        return f"❌ [Sim {sim_id}] Model Load Error: {e}"
    
    try:
        src_dir = os.path.join(RAW_DATA_ROOT, mode, f"sim_{sim_id}")
        T_seed = torch.load(os.path.join(src_dir, "e1_Tprev_data_select_snaps.pt"), map_location=device)[0:1].double()
        u_true_0 = torch.load(os.path.join(src_dir, "e1_uprev_data_select_snaps.pt"), map_location=device)[0:1].double()
        v_true_0 = torch.load(os.path.join(src_dir, "e1_vprev_data_select_snaps.pt"), map_location=device)[0:1].double()
    except: return f"❌ [Sim {sim_id}] Failed to load seed data from {src_dir}"

    t0_sim = time.time()
    print(f"🚀 [PID {os.getpid()}] Sim {sim_id} Start. Q: {q_val:.2f}")

    inputs = prepare_inputs(T_seed, sim_info, device)
    with torch.no_grad():
        _, _, u_raw, v_raw, _, _ = teacher(*inputs)
        raw_max = torch.max(torch.sqrt(u_raw**2 + v_raw**2)).item()
        true_max = torch.max(torch.sqrt(u_true_0**2 + v_true_0**2)).item()
        scaler = min(true_max / raw_max, SCALER_CAP) if raw_max > 1e-4 else 1.0

    u_curr, v_curr, p_curr, _ = project_velocity_mac(u_true_0, v_true_0, DX, DY, device, p_prev=None)
    T_curr = T_seed
    
    data_cache = {
        'T': [T_curr.cpu().double()], 
        'u': [u_curr.cpu().double()], 
        'v': [v_curr.cpu().double()],
        'p': [p_curr.cpu().double()], 
        'div': [torch.zeros_like(p_curr).cpu().double()],
        'u_star': [u_curr.cpu().double()], 
        'v_star': [v_curr.cpu().double()]
    }

    for step in range(1, STEPS_PER_SIM + 1):
        with torch.no_grad():
            inputs[0] = T_curr
            _, _, u_raw, v_raw, _, _ = teacher(*inputs)
            
            u_phys = INERTIA_ALPHA * u_curr + (1 - INERTIA_ALPHA) * u_raw * scaler
            v_phys = INERTIA_ALPHA * v_curr + (1 - INERTIA_ALPHA) * v_raw * scaler
            
            d_u = torch.clamp(u_phys - u_curr, -0.05*torch.abs(u_curr)-5.0, 0.05*torch.abs(u_curr)+5.0)
            d_v = torch.clamp(v_phys - v_curr, -0.05*torch.abs(v_curr)-5.0, 0.05*torch.abs(v_curr)+5.0)
            u_phys = u_curr + d_u
            v_phys = v_curr + d_v
            
            u_star_val, v_star_val = u_phys.clone(), v_phys.clone()
            
            u_phys, v_phys, p_new, div_in_val = project_velocity_mac(u_phys, v_phys, DX, DY, device, p_prev=p_curr)
            p_curr = p_new
            
            T_next = rk4_step_adaptive(T_curr, u_phys, v_phys, DT_PHYS, DX, DY, q_val, TEMP_AT_IDX_0, TEMP_AT_IDX_LAST)
            T_curr = T_next.double()
            u_curr = u_phys.double(); v_curr = v_phys.double()

            if step % SAVE_INTERVAL == 0:
                data_cache['T'].append(T_next.cpu().double())
                data_cache['u'].append(u_phys.cpu().double())
                data_cache['v'].append(v_phys.cpu().double())
                data_cache['p'].append(p_new.cpu().double())
                data_cache['div'].append(div_in_val.cpu().double())
                data_cache['u_star'].append(u_star_val.cpu().double())
                data_cache['v_star'].append(v_star_val.cpu().double())

            if step % 200 == 0: 
                print(f"   [Sim {sim_id}] Step {step}/{STEPS_PER_SIM}")

    os.makedirs(target_dir, exist_ok=True)
    torch.save(torch.cat(data_cache['T']), os.path.join(target_dir, "e1_Tprev_data_select_snaps.pt"))
    torch.save(torch.cat(data_cache['u']), os.path.join(target_dir, "e1_uprev_data_select_snaps.pt"))
    torch.save(torch.cat(data_cache['v']), os.path.join(target_dir, "e1_vprev_data_select_snaps.pt"))
    torch.save(torch.cat(data_cache['p']), os.path.join(target_dir, "e1_pprev_data_select_snaps.pt"))
    torch.save(torch.cat(data_cache['div']), os.path.join(target_dir, "e1_div_in_data_select_snaps.pt"))
    torch.save(torch.cat(data_cache['u_star']), os.path.join(target_dir, "e1_ustar_data_select_snaps.pt"))
    torch.save(torch.cat(data_cache['v_star']), os.path.join(target_dir, "e1_vstar_data_select_snaps.pt"))
    
    return f"✅ [Sim {sim_id}] Done in {time.time() - t0_sim:.1f}s"

# ==========================================
# 5. Main Entry Point
# ==========================================
if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    
    os.makedirs(DATA_ROOT, exist_ok=True)
    
    try:
        sims = torch.load(os.path.join(RAW_DATA_ROOT, "sims.pt"), weights_only=False)
        torch.save(sims, os.path.join(DATA_ROOT, "sims.pt"))
    except FileNotFoundError:
        print(f"❌ Error: 'sims.pt' not found in {RAW_DATA_ROOT}")
        print("   Did you move the 'TPH' folder into the 'teacher' directory?")
        sys.exit(1)
    
    print(f"🛡️  ULTRA-PRECISION GENERATOR (Modularized)")
    print(f"📂  Teacher Dir: {TEACHER_DIR}")
    print(f"📂  Target Output: {DATA_ROOT}")
    
    NUM_WORKERS = 8
    tasks = [(i, sims[i], "cuda" if torch.cuda.is_available() else "cpu") for i in range(len(sims))]
    
    ctx = mp.get_context('spawn')
    with ctx.Pool(NUM_WORKERS) as pool:
        for result in pool.imap_unordered(process_simulation, tasks):
            print(result)

    print("🎉 All simulations completed.")