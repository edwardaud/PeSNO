import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import os
import random
from configs.config import H_GRID, W_GRID, DX, VISCOSITY_SCALE_FACTOR

class AugmentedStreamDataset(Dataset):
    def __init__(self, root_dir, mode='train', limit_size=None, probe=False, 
                 external_stats=None):        
        self.root_dir = root_dir
        self.mode = mode
        all_sims = torch.load(os.path.join(root_dir, "sims.pt"), weights_only=False)
        
        # 1. Filter samples
        raw_indices = [i for i, s in enumerate(all_sims) if s[1] == mode and s[0] not in [17, 87]]
        if limit_size: raw_indices = raw_indices[:limit_size]
        
        self.sim_indices = self._sanity_check(all_sims, raw_indices)
        
        # 2. Stats Probing (Critical for Normalization)
        if probe and mode == 'train':
            self.psi_norm, self.dtdx_norm = self._run_probe(all_sims, raw_indices)
        else:
            if external_stats is None:
                # Default safety or raise error
                print(f"⚠️ [{mode.upper()}] Warning: No external stats provided, using default 1.0")
                self.psi_norm, self.dtdx_norm = 1.0, 1.0
            else:
                self.psi_norm = external_stats['psi_norm']
                self.dtdx_norm = external_stats['dtdx_norm']
                print(f"[{mode.upper()}] Loaded Stats: PSI={self.psi_norm:.4f}, DTDX={self.dtdx_norm:.4f}")

        # 3. Grid Generation [X, Y]
        gy, gx = torch.meshgrid(torch.linspace(0, 1, H_GRID), torch.linspace(0, 4, W_GRID), indexing='ij')
        self.grid = torch.stack((gx, gy), dim=0).float() 
        
        self.data_list = []
        self.sim_data_cache = {}
        
        print(f"[{mode.upper()}] Indexing dataset...")
        for sim_idx in self.sim_indices:
            self._load_sim_to_cache(all_sims, sim_idx)
            
        print(f"[{mode.upper()}] Ready. Total samples: {len(self.data_list)}")

    def _load_sim_to_cache(self, all_sims, sim_idx):
        s = all_sims[sim_idx]
        sub = os.path.join(self.root_dir, s[1], f"sim_{s[0]}")
        try:
            # Load Tensors (CPU)
            T = torch.load(os.path.join(sub, "e1_Tprev_data_select_snaps.pt"), map_location='cpu').float()
            psi = torch.load(os.path.join(sub, "e1_psi_data_select_snaps.pt"), map_location='cpu').float()
            u = torch.load(os.path.join(sub, "e1_uprev_data_select_snaps.pt"), map_location='cpu').float()
            v = torch.load(os.path.join(sub, "e1_vprev_data_select_snaps.pt"), map_location='cpu').float()
            
            if T.dim() == 4: T = T.squeeze(1)
            if psi.dim() == 4: psi = psi.squeeze(1)
            if u.dim() == 4: u = u.squeeze(1)
            if v.dim() == 4: v = v.squeeze(1)
            
            p_vec = torch.tensor([(s[2]-5.0)/5.0, (np.log10(s[3]+1e-9)-6.0)/4.0, (np.log10(s[4]+1e-9)-0.0)/2.0], dtype=torch.float32)
            log_c, log_b = float(np.log(s[3]+1e-9)), float(np.log(s[4]+1e-9))
            
            self.sim_data_cache[sim_idx] = {'T': T, 'psi': psi, 'u': u, 'v': v, 'p_vec': p_vec, 'log_c': log_c, 'log_b': log_b}
            
            for t in np.linspace(0, T.shape[0]-1, 1000, dtype=int):
                self.data_list.append((sim_idx, t))
        except Exception:
            pass

    def _sanity_check(self, all_sims, indices):
        valid = []
        for idx in indices:
            path = os.path.join(self.root_dir, all_sims[idx][1], f"sim_{all_sims[idx][0]}", "e1_psi_data_select_snaps.pt")
            if os.path.exists(path): valid.append(idx)
        return valid

    def _run_probe(self, all_sims, indices):
        print("📊 [Auto-Probe] Scanning Statistics...")
        psi_vals, dtdx_vals = [], []
        sample_indices = random.sample(indices, min(20, len(indices)))
        for idx in sample_indices:
            s = all_sims[idx]
            try:
                psi = torch.load(os.path.join(self.root_dir, s[1], f"sim_{s[0]}", "e1_psi_data_select_snaps.pt"), map_location='cpu')
                psi_vals.append(psi.abs())
                T = torch.load(os.path.join(self.root_dir, s[1], f"sim_{s[0]}", "e1_Tprev_data_select_snaps.pt"), map_location='cpu')
                T_pad = F.pad(T, (1, 1, 0, 0), mode='circular')
                dtdx = (T_pad[..., 2:] - T_pad[..., :-2]) / (2 * DX)
                dtdx_vals.append(dtdx.abs())
            except: pass
        
        psi_norm = np.percentile(torch.cat(psi_vals).numpy(), 99.9)
        dtdx_norm = np.percentile(torch.cat(dtdx_vals).numpy(), 99.99)
        print(f"✅ Auto-Config: PSI_NORM={psi_norm:.4f}, DTDX_NORM={dtdx_norm:.4f}")
        return psi_norm, dtdx_norm

    def __len__(self): return len(self.data_list)

    def __getitem__(self, idx):
        sim_idx, t = self.data_list[idx]
        data = self.sim_data_cache[sim_idx]
        
        T_frame = data['T'][t].unsqueeze(0)
        psi_frame = data['psi'][t].unsqueeze(0)
        u_frame = data['u'][t].unsqueeze(0)
        v_frame = data['v'][t].unsqueeze(0)
        
        T_pad = F.pad(T_frame, (1, 1, 0, 0), mode='circular')
        dTdx = (T_pad[..., 2:] - T_pad[..., :-2]) / (2 * DX)
        dTdx_norm = dTdx / (self.dtdx_norm + 1e-6) 
        
        log_eta = (-data['log_c']*T_frame + data['log_b']*(1.0 - self.grid[1:2])) / VISCOSITY_SCALE_FACTOR
        p_map = data['p_vec'].view(3, 1, 1).expand(3, H_GRID, W_GRID)
        
        x = torch.cat([T_frame, self.grid, p_map, log_eta, dTdx_norm], dim=0)
        y_norm = psi_frame / self.psi_norm
        
        return x, y_norm, data['p_vec'], u_frame, v_frame