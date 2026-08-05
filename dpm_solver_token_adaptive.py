import torch
import numpy as np
from dpm_solver_pytorch import *

class TokenAdaptiveDPM_Solver:
    def __init__(
        self,
        model_fn,             # callable: (x: BxLxd, t_ids: BxL) -> pred_x0: BxLxd
        alphas_cumprod_2d,    # torch.Tensor shape (T, L), on device
        algorithm_type="dpmsolver++",
        correcting_x0_fn=None,
    ):
        self.model_fn = model_fn
        self.alphas_cumprod_2d = alphas_cumprod_2d   # (T, L)
        self.algorithm_type = algorithm_type
        self.correcting_x0_fn = correcting_x0_fn

    def _token_schedule_coeffs(self, t_ids):
        # t_ids: (B, L) or (L,) long tensor. 
        if t_ids.dim() == 1:
            L = t_ids.shape[0]
            ac = self.alphas_cumprod_2d               # (T, L)
            ac_t = ac[t_ids, torch.arange(L, device=t_ids.device)]  # (L,)
            alpha_t = torch.sqrt(ac_t)
            sigma_t = torch.sqrt(1.0 - ac_t)
            lambda_t = 0.5 * torch.log(ac_t / (1.0 - ac_t + 1e-9))  # half-logSNR
            return alpha_t, sigma_t, lambda_t
        else: # (B, L)
            B, L = t_ids.shape
            ac = self.alphas_cumprod_2d               # (T, L)
            i_idx = torch.arange(L, device=t_ids.device).unsqueeze(0).expand(B, L)
            ac_t = ac[t_ids, i_idx] # (B, L)
            alpha_t = torch.sqrt(ac_t)
            sigma_t = torch.sqrt(1.0 - ac_t)
            lambda_t = 0.5 * torch.log(ac_t / (1.0 - ac_t + 1e-9))  # half-logSNR
            return alpha_t, sigma_t, lambda_t

    def _token_first_update(self, x, t_s_ids, t_t_ids, model_s=None):
        """
        x          : (B, L, d)
        t_s_ids    : (L,) or (B,L)  source timestep per token
        t_t_ids    : (L,) or (B,L) target timestep per token
        model_s    : (B, L, d) or None -- predicted x0 at source; computed if None
        """
        alpha_s, sigma_s, lambda_s = self._token_schedule_coeffs(t_s_ids)  # (L,) or (B,L)
        alpha_t, sigma_t, lambda_t = self._token_schedule_coeffs(t_t_ids)  # (L,) or (B,L)
        h     = lambda_t - lambda_s   # (L,) or (B,L)
        phi_1 = torch.expm1(-h)       # (L,) or (B,L)

        if model_s is None:
            model_s = self.model_fn(x, t_s_ids)   # (B, L, d)
            if self.correcting_x0_fn is not None:
                model_s = self.correcting_x0_fn(model_s)

        if h.dim() == 1:
            def bcast(v): return v[None, :, None]
        else:
            def bcast(v): return v[:, :, None]

        x_t = (
            bcast(sigma_t / sigma_s) * x
            - bcast(alpha_t * phi_1) * model_s
        )
        return x_t, model_s

    def _token_multistep_second_update(self, x, model_prev_list, t_prev_ids_list, t_t_ids):
        m0 = model_prev_list[-1]      # (B, L, d)
        m1 = model_prev_list[-2]      # (B, L, d)
        t0_ids = t_prev_ids_list[-1]  # (L,) or (B,L)
        t1_ids = t_prev_ids_list[-2]  # (L,) or (B,L)

        _, _,       lam_prev1   = self._token_schedule_coeffs(t1_ids)
        alpha_prev0, sigma_prev0, lam_prev0 = self._token_schedule_coeffs(t0_ids)
        alpha_t,    sigma_t,    lam_t       = self._token_schedule_coeffs(t_t_ids)

        h_0 = lam_prev0 - lam_prev1  
        h   = lam_t     - lam_prev0  
        r0  = h_0 / h                 

        if r0.dim() == 1:
            D1_0 = (1.0 / r0)[None, :, None] * (m0 - m1)  # (B, L, d)
        else:
            D1_0 = (1.0 / r0)[:, :, None] * (m0 - m1)

        phi_1 = torch.expm1(-h)       
        if h.dim() == 1:
            def bcast(v): return v[None, :, None]
        else:
            def bcast(v): return v[:, :, None]

        x_t = (
            bcast(sigma_t / sigma_prev0) * x
            - bcast(alpha_t * phi_1) * m0
            - 0.5 * bcast(alpha_t * phi_1) * D1_0
        )
        return x_t

    def sample(
        self,
        x,                  
        J,                  
        order=2,
        x_start=None,       
        input_ids_mask=None,
    ):
        """
        K-1 solver steps. Step r moves token i from J[r, i] to J[r+1, i].
        J must have rows in descending t order (noisiest first).
        """
        K = J.shape[0]  

        model_prev_list = []
        t_prev_ids_list = []

        for r in range(K - 1):
            t_s_ids = J[r]       
            t_t_ids = J[r + 1]   

            if r == 0 or order == 1 or len(model_prev_list) < 2:
                x, model_s = self._token_first_update(x, t_s_ids, t_t_ids)
            else:
                model_s = self.model_fn(x, t_s_ids)
                if self.correcting_x0_fn is not None:
                    model_s = self.correcting_x0_fn(model_s)
                x = self._token_multistep_second_update(
                    x, model_prev_list, t_prev_ids_list, t_t_ids
                )

            model_prev_list.append(model_s)
            t_prev_ids_list.append(t_s_ids)
            if len(model_prev_list) > order:
                model_prev_list.pop(0)
                t_prev_ids_list.pop(0)

            if input_ids_mask is not None and x_start is not None:
                x = torch.where(input_ids_mask == 0, x_start, x)

        return x

@torch.no_grad()
def build_token_timestep_matrix(model, diffusion, data_loader, K, device,
                                 T_subset=None, smooth_sigma=5.0,
                                 profile_batches=10):
    from scipy.ndimage import gaussian_filter1d
    from scipy.integrate import cumulative_trapezoid

    T = diffusion.num_timesteps
    L = diffusion.token_max_length
    ac = diffusion.alphas_cumprod  # (T, L) numpy

    t_range = np.arange(T) if T_subset is None else np.random.choice(T, T_subset, replace=False)
    E_sum   = np.zeros((T, L), dtype=np.float64)
    E_count = np.zeros((T, L), dtype=np.float64)

    for batch_idx, (z0, cond) in enumerate(data_loader):
        if batch_idx >= profile_batches:
            break
        z0 = z0.to(device)                          # (B, L, d)
        B = z0.shape[0]
        
        input_ids_mask = cond['input_mask'] # (B, L)
        input_ids_mask_3d = input_ids_mask.unsqueeze(-1).expand_as(z0).to(device)
        
        for t_int in t_range:
            t_tensor = torch.full((B,), t_int, device=device, dtype=torch.long)
            t_tensor_2d = t_tensor.unsqueeze(-1).expand(B, L)
            
            noise = torch.randn_like(z0)
            z_t = diffusion.q_sample(z0, t_tensor)  # (B, L, d)
            z_t = torch.where(input_ids_mask_3d == 0, z0, z_t)
            
            pred_x0 = model(z_t, t_tensor_2d)  # (B, L, d)
            loss = ((pred_x0 - z0) ** 2).mean(-1)   # (B, L) 
            loss = loss * input_ids_mask.to(device)
            E_sum[t_int]   += loss.sum(0).cpu().numpy()
            E_count[t_int] += B

    E_count = np.where(E_count == 0, 1.0, E_count)
    E = E_sum / E_count                              # (T, L)

    J = np.zeros((K, L), dtype=np.int64)
    lambda_array = 0.5 * np.log(ac / (1.0 - ac + 1e-9))  # (T, L) 

    for i in range(L):
        lambda_i = lambda_array[:, i]   
        E_i      = E[:, i]              

        if smooth_sigma > 0:
            E_i_smooth = gaussian_filter1d(E_i, sigma=smooth_sigma)
        else:
            E_i_smooth = E_i.copy()

        d_lam  = np.diff(lambda_i)             
        d_E    = np.diff(E_i_smooth)           
        d_lam_safe = np.where(np.abs(d_lam) < 1e-9, 1e-9, d_lam)
        slope  = d_E / d_lam_safe              

        lam_mid = (lambda_i[:-1] + lambda_i[1:]) / 2.0  

        a_i = -np.exp(lam_mid) * slope
        a_i = np.clip(a_i, 0, None)            

        G_i = cumulative_trapezoid(a_i, lam_mid, initial=0.0)  
        G_max = G_i[-1]

        if G_max < 1e-9:
            lam_star_i = np.linspace(lam_mid[0], lam_mid[-1], K)
        else:
            G_norm = G_i / G_max                    
            q      = np.linspace(0.0, 1.0, K)
            lam_star_i = np.interp(q, G_norm, lam_mid)  

        for r in range(K):
            J[r, i] = np.argmin(np.abs(lambda_i - lam_star_i[r]))

    J = np.sort(J, axis=0)[::-1].copy()  
    return J  

def token_model_wrapper(model, alphas_cumprod_2d, model_kwargs={}):
    """
    Returns: model_fn(x: BxLxd, t_ids: L long or BxL long) -> pred_x0: BxLxd
    Uses the per-token t_ids as the 2D timestep for the transformer (Option B).
    """
    def model_fn(x, t_ids):
        # Expand t_ids to (B, L) if it is (L,)
        if t_ids.dim() == 1:
            t_batch = t_ids.unsqueeze(0).expand(x.shape[0], -1) # (B, L)
        else:
            t_batch = t_ids
        pred_x0  = model(x, t_batch, **model_kwargs)
        return pred_x0
    return model_fn
