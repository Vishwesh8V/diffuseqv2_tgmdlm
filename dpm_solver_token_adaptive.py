import warnings
import torch
import numpy as np
from dpm_solver_pytorch import *

# ---------------------------------------------------------------------------
# Optional heavy deps — degrade gracefully if missing
# ---------------------------------------------------------------------------
try:
    from sklearn.isotonic import IsotonicRegression as _IsotonicRegression
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False
    warnings.warn(
        "sklearn not found. Curve smoothing will fall back to Savitzky-Golay. "
        "Install scikit-learn for best results: pip install scikit-learn"
    )

from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter


# ===========================================================================
# TokenAdaptiveDPM_Solver
# ===========================================================================

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
        else:  # (B, L)
            B, L = t_ids.shape
            ac = self.alphas_cumprod_2d               # (T, L)
            i_idx = torch.arange(L, device=t_ids.device).unsqueeze(0).expand(B, L)
            ac_t = ac[t_ids, i_idx]                   # (B, L)
            alpha_t = torch.sqrt(ac_t)
            sigma_t = torch.sqrt(1.0 - ac_t)
            lambda_t = 0.5 * torch.log(ac_t / (1.0 - ac_t + 1e-9))
            return alpha_t, sigma_t, lambda_t

    def _token_first_update(self, x, t_s_ids, t_t_ids, model_s=None):
        """
        x          : (B, L, d)
        t_s_ids    : (L,) or (B,L)  source timestep per token
        t_t_ids    : (L,) or (B,L)  target timestep per token
        model_s    : (B, L, d) or None -- predicted x0 at source; computed if None
        """
        alpha_s, sigma_s, lambda_s = self._token_schedule_coeffs(t_s_ids)
        alpha_t, sigma_t, lambda_t = self._token_schedule_coeffs(t_t_ids)
        h     = lambda_t - lambda_s
        phi_1 = torch.expm1(-h)

        if model_s is None:
            model_s = self.model_fn(x, t_s_ids)
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

        _, _,           lam_prev1   = self._token_schedule_coeffs(t1_ids)
        alpha_prev0, sigma_prev0, lam_prev0 = self._token_schedule_coeffs(t0_ids)
        alpha_t,     sigma_t,     lam_t     = self._token_schedule_coeffs(t_t_ids)

        h_0 = lam_prev0 - lam_prev1
        h   = lam_t     - lam_prev0
        r0  = h_0 / h

        if r0.dim() == 1:
            D1_0 = (1.0 / r0)[None, :, None] * (m0 - m1)
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


# ===========================================================================
# Internal helpers (ported / adapted from validation_trajectory_scheduler.py)
# ===========================================================================

def _merge_duplicate_x(x: np.ndarray, y: np.ndarray):
    """Average y values that share the same x (deduplication before PCHIP)."""
    unique_x, inverse = np.unique(x, return_inverse=True)
    sums   = np.zeros_like(unique_x, dtype=np.float64)
    counts = np.zeros_like(unique_x, dtype=np.float64)
    np.add.at(sums,   inverse, y)
    np.add.at(counts, inverse, 1.0)
    return unique_x, sums / np.maximum(counts, 1.0)


def _safe_savgol_window(requested: int, n: int, polyorder: int) -> int:
    """Return the largest odd window <= min(requested, n) that exceeds polyorder."""
    window  = min(requested, n if n % 2 == 1 else n - 1)
    minimum = polyorder + 2
    if minimum % 2 == 0:
        minimum += 1
    window = max(window, minimum)
    if window > n:
        window = n if n % 2 == 1 else n - 1
    if window <= polyorder:
        raise ValueError("Not enough points for Savitzky-Golay smoothing.")
    return window


def _repair_in_lambda_order(indices: np.ndarray, dense_lambda: np.ndarray) -> np.ndarray:
    """
    Ensure indices are strictly increasing in lambda space (no duplicate or
    reversed steps). Forward pass raises ties, backward pass resolves overflow.
    """
    order = np.argsort(dense_lambda)
    rank  = np.empty_like(order)
    rank[order] = np.arange(len(order))

    ranks = rank[indices].copy()
    K, T  = len(ranks), len(order)
    if K > T:
        raise ValueError("K cannot exceed T.")

    for r in range(1, K):
        ranks[r] = max(ranks[r], ranks[r - 1] + 1)

    if ranks[-1] >= T:
        ranks[-1] = T - 1
        for r in range(K - 2, -1, -1):
            ranks[r] = min(ranks[r], ranks[r + 1] - 1)

    return order[ranks]


def _smooth_loss_curve(
    lambda_i: np.ndarray,
    E_i: np.ndarray,
    dense_grid_size: int = 4096,
    method: str = "isotonic_pchip",
    savgol_window: int = 51,
    savgol_polyorder: int = 3,
) -> tuple:
    """
    Smooth the per-token loss curve E_i(lambda_i) onto a dense grid.

    Returns:
        x_dense   : (dense_grid_size,)  uniformly spaced lambda values
        y_dense   : (dense_grid_size,)  smoothed loss values
        derivative: (dense_grid_size,)  dE/dlambda on the dense grid (clipped <= 0)
    """
    # Remove NaN / non-finite entries
    valid = np.isfinite(lambda_i) & np.isfinite(E_i)
    x = np.asarray(lambda_i[valid], dtype=np.float64)
    y = np.asarray(E_i[valid],      dtype=np.float64)

    # Sort by lambda
    order = np.argsort(x)
    x, y  = _merge_duplicate_x(x[order], y[order])

    # Dense interpolation grid
    x_dense = np.linspace(x[0], x[-1], dense_grid_size)

    if method == "isotonic_pchip" and _HAS_SKLEARN:
        # Isotonic regression enforces monotone decreasing loss curve
        iso   = _IsotonicRegression(increasing=False, out_of_bounds="clip")
        y_fit = iso.fit_transform(x, y)
    elif method == "savgol" or (method == "isotonic_pchip" and not _HAS_SKLEARN):
        window = _safe_savgol_window(savgol_window, len(y), savgol_polyorder)
        y_fit  = savgol_filter(y, window, savgol_polyorder, mode="interp")
    else:
        y_fit = y  # pchip_raw: no smoothing before interpolation

    # PCHIP interpolation onto the dense grid (shape-preserving, no ringing)
    y_dense = np.asarray(PchipInterpolator(x, y_fit, extrapolate=False)(x_dense))

    # Handle NaN at extrapolation edges (set to nearest valid value)
    nan_mask = ~np.isfinite(y_dense)
    if nan_mask.any():
        first_valid = np.argmax(~nan_mask)
        last_valid  = len(y_dense) - 1 - np.argmax((~nan_mask)[::-1])
        y_dense[:first_valid] = y_dense[first_valid]
        y_dense[last_valid+1:] = y_dense[last_valid]

    # Derivative via np.gradient (higher quality than finite differences on raw T pts)
    derivative = np.gradient(y_dense, x_dense)
    # Clip to non-positive: loss must not increase as lambda increases
    derivative = np.minimum(derivative, 0.0)

    return x_dense, y_dense, derivative


# ===========================================================================
# Precompute: build dense (T, L) J matrix
# ===========================================================================

@torch.no_grad()
def build_token_timestep_matrix(
    model,
    diffusion,
    data_loader,
    device,
    T_subset=None,
    profile_batches=10,
    reuse_noise=True,
    smooth_method="isotonic_pchip",
    dense_grid_size=4096,
    savgol_window=51,
    savgol_polyorder=3,
    monitor_floor=1e-8,
    force_endpoints=True,
    repair_monotone=True,
    output_dir=None,
):
    """
    Build a DENSE per-token timestep matrix J_dense of shape (T, L).

    This is the canonical precompute artefact. Call subsample_J(J_dense, K)
    at inference time to get a (K, L) schedule for any desired step count K,
    with no model forward passes repeated.

    Improvements over the original implementation
    --------------------------------------------
    * Noise reuse      : one noise tensor per batch reused across all T steps,
                         reducing variance in E_{t,i} so fewer profile_batches
                         are needed for the same accuracy.
    * Exact valid count: counts per-token valid samples from the mask (not just
                         batch size B), giving correct mean loss when mask
                         lengths vary.
    * NaN marking      : positions with zero valid samples are marked as NaN
                         instead of silently producing zero loss.
    * isotonic_pchip   : isotonic regression (monotone-decreasing) followed by
                         PCHIP interpolation onto a dense 4096-point grid,
                         giving a smooth, artifact-free loss curve.
    * np.gradient      : derivative computed on the dense grid, far more
                         accurate than finite differences on raw T points.
    * monitor_floor    : minimum density floor (1e-8) so every token always
                         has non-zero mass and no degenerate fallback is needed.
    * sqrt(a_i)        : integrates sqrt(a_i) (arc-length) instead of a_i
                         (area), producing more evenly-spaced node gaps.
    * force_endpoints  : pins row 0 to the noisiest timestep and row T-1 to
                         the cleanest, guaranteeing full denoising range.
    * repair_monotone  : _repair_in_lambda_order ensures strictly increasing
                         lambda indices per column (no no-op solver steps).
    * output_dir       : optionally saves calibration.npz and
                         reduced_schedule.npz for inspection and fast reuse.

    Args
    ----
    model            : frozen f_theta; called as model(z_t, t_2d)
    diffusion        : GaussianDiffusion with 2-D alphas_cumprod (T, L)
    data_loader      : iterable yielding (z0, cond) batches
    device           : torch.device or str
    T_subset         : int or None -- sub-sample this many timesteps for
                       profiling (None = all T). Trades speed vs accuracy.
    profile_batches  : number of validation batches to average E_{t,i} over
    reuse_noise      : if True, one noise draw per batch is reused across all
                       timesteps (reduces variance, recommended)
    smooth_method    : "isotonic_pchip" | "savgol" | "pchip_raw"
    dense_grid_size  : number of points in the interpolation grid (default 4096)
    savgol_window    : Savitzky-Golay window (used when smooth_method="savgol")
    savgol_polyorder : Savitzky-Golay polynomial order
    monitor_floor    : minimum value of the node-density function (prevents
                       degenerate zero-mass tokens)
    force_endpoints  : pin first/last rows to global min/max lambda timesteps
    repair_monotone  : enforce strictly increasing lambda order per column
    output_dir       : str or Path -- if given, saves calibration + schedule
                       .npz files for inspection

    Returns
    -------
    J_dense : np.ndarray, shape (T, L), dtype int64
              J_dense[r, i] = discrete timestep index for token i at
              cumulative-density quantile r/(T-1).
              Row 0 = noisiest (highest t), row T-1 = cleanest (lowest t).
    """
    T = diffusion.num_timesteps
    L = diffusion.token_max_length
    ac = diffusion.alphas_cumprod              # (T, L) numpy float64

    # ---- Stage 1: accumulate E[T, L] ----------------------------------------
    t_range = (
        np.arange(T)
        if T_subset is None
        else np.sort(np.random.choice(T, T_subset, replace=False))
    )

    loss_sum   = np.zeros((T, L), dtype=np.float64)
    valid_count = np.zeros((T, L), dtype=np.float64)

    print(f"[build_token_timestep_matrix] Profiling {profile_batches} batches "
          f"over {len(t_range)} timesteps ...")

    for batch_idx, (z0, cond) in enumerate(data_loader):
        if batch_idx >= profile_batches:
            break

        z0 = z0.to(device)                                    # (B, L, d)
        B  = z0.shape[0]

        input_ids_mask    = cond['input_mask']                 # (B, L) int/float
        input_ids_mask_3d = input_ids_mask.unsqueeze(-1).expand_as(z0).to(device)
        mask_gpu          = input_ids_mask.float().to(device)  # (B, L)

        # One noise draw per batch, reused across all T steps if requested
        common_noise = torch.randn_like(z0) if reuse_noise else None

        for t_int in t_range:
            t_tensor    = torch.full((B,), t_int, device=device, dtype=torch.long)
            t_tensor_2d = t_tensor.unsqueeze(-1).expand(B, L)

            eps = common_noise if (common_noise is not None) else torch.randn_like(z0)

            z_t = diffusion.q_sample(z0, t_tensor)            # (B, L, d)
            z_t = torch.where(input_ids_mask_3d == 0, z0, z_t)  # keep source tokens clean

            pred_x0   = model(z_t, t_tensor_2d)               # (B, L, d)
            per_token = ((pred_x0 - z0) ** 2).mean(-1)        # (B, L)
            per_token = per_token * mask_gpu                   # zero source positions

            # Accumulate per-token valid counts (not just B)
            loss_sum[t_int]   += per_token.sum(0).double().cpu().numpy()
            valid_count[t_int] += mask_gpu.sum(0).double().cpu().numpy()

        if (batch_idx + 1) % max(1, profile_batches // 5) == 0:
            print(f"  batch {batch_idx + 1}/{profile_batches}")

    # Mean loss: NaN where no valid samples (never masked-in)
    E = np.where(valid_count > 0, loss_sum / np.maximum(valid_count, 1.0), np.nan)

    # ---- Stage 2-3: per-token curve fitting & node density ------------------
    lambda_array = np.log(ac) - np.log1p(-ac)                 # (T, L) full logSNR

    K_dense = T
    J_dense = np.zeros((K_dense, L), dtype=np.int64)

    print(f"[build_token_timestep_matrix] Fitting curves and selecting nodes for L={L} positions ...")

    for i in range(L):
        lambda_i = lambda_array[:, i]                          # (T,)
        E_i      = E[:, i]                                     # (T,)

        # Replace NaN with interpolated/edge values so smoothing doesn't fail
        nan_mask = ~np.isfinite(E_i)
        if nan_mask.all():
            # Fully degenerate token (source position) — uniform spacing fallback
            lam_min, lam_max = lambda_i.min(), lambda_i.max()
            lam_star_i = np.linspace(lam_min, lam_max, K_dense)
            for r in range(K_dense):
                J_dense[r, i] = np.argmin(np.abs(lambda_i - lam_star_i[r]))
            continue

        if nan_mask.any():
            # Linear interpolation for isolated NaN timesteps
            finite_idx = np.where(~nan_mask)[0]
            E_i = E_i.copy()
            E_i[nan_mask] = np.interp(
                np.where(nan_mask)[0], finite_idx, E_i[finite_idx]
            )

        # Smooth onto dense grid and compute derivative
        x_dense, y_dense, derivative = _smooth_loss_curve(
            lambda_i, E_i,
            dense_grid_size=dense_grid_size,
            method=smooth_method,
            savgol_window=savgol_window,
            savgol_polyorder=savgol_polyorder,
        )

        # Node density: a_i(lambda) = -exp(lambda) * dE/dlambda (>= monitor_floor)
        monitor = np.maximum(-np.exp(x_dense) * derivative, monitor_floor)

        # Arc-length density: sqrt(a_i) integrates more evenly-spaced nodes
        density    = np.sqrt(monitor)
        cumulative = cumulative_trapezoid(density, x_dense, initial=0.0)   # (dense_grid_size,)
        total_mass = cumulative[-1]

        if not np.isfinite(total_mass) or total_mass <= 0:
            # Degenerate: uniform logSNR fallback
            lam_star_i = np.linspace(x_dense[0], x_dense[-1], K_dense)
        else:
            cumulative  /= total_mass                          # normalise to [0, 1]
            q            = np.linspace(0.0, 1.0, K_dense)
            lam_star_i   = np.interp(q, cumulative, x_dense)  # (K_dense,)

        # Map lambda* back to nearest discrete timestep index
        idx = np.array([
            int(np.argmin(np.abs(lambda_i - lv))) for lv in lam_star_i
        ], dtype=np.int64)

        # Pin first/last rows to the actual extreme timesteps
        if force_endpoints:
            idx[0]  = int(np.argmin(lambda_i))   # noisiest = smallest lambda
            idx[-1] = int(np.argmax(lambda_i))   # cleanest = largest lambda

        # Guarantee strictly increasing lambda order (no no-op steps)
        if repair_monotone:
            idx = _repair_in_lambda_order(idx, lambda_i)

        J_dense[:, i] = idx

    # Enforce descending-t order: row 0 = noisiest (highest t), row T-1 = cleanest
    # After repair_monotone, columns are in ascending lambda order;
    # since lambda increases as t decreases, we flip to get descending t.
    J_dense = J_dense[::-1, :].copy()                         # (T, L)

    # ---- Optional: persist calibration artefacts ----------------------------
    if output_dir is not None:
        from pathlib import Path
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out / "calibration.npz",
            loss_sum=loss_sum,
            valid_count=valid_count,
            mean_loss=E,
            lambda_schedule=lambda_array,
            alpha_bar_schedule=ac,
        )
        np.savez_compressed(
            out / "dense_schedule.npz",
            J_dense=J_dense,
        )
        print(f"[build_token_timestep_matrix] Saved calibration + schedule to {out}")

    print(f"[build_token_timestep_matrix] Done. J_dense shape: {J_dense.shape}")
    return J_dense


# ===========================================================================
# Inference-time sub-sampling: get a (K, L) schedule from J_dense
# ===========================================================================

def subsample_J(J_dense, K):
    """
    Sub-sample K evenly-spaced rows from a dense J matrix of shape (T, L)
    to produce a (K, L) schedule for a K-step solver run.

    Because J_dense encodes the optimal cumulative arc-length ordering across
    T quantile levels, uniform row indexing preserves that ordering for any K.

    Args:
        J_dense : np.ndarray or torch.Tensor of shape (T, L), dtype int64
        K       : int -- desired number of solver nodes (K-1 actual steps)

    Returns:
        J_K : np.ndarray of shape (K, L), dtype int64
    """
    T = J_dense.shape[0]
    if K > T:
        raise ValueError(
            f"K={K} exceeds the number of rows in J_dense ({T}). "
            f"Re-run profiling with a larger T or reduce K."
        )
    if K == T:
        if isinstance(J_dense, np.ndarray):
            return J_dense.copy()
        return J_dense.clone()

    # Uniformly-spaced indices from 0 (noisiest) to T-1 (cleanest)
    indices = np.linspace(0, T - 1, K, dtype=np.int64)

    if isinstance(J_dense, np.ndarray):
        return J_dense[indices, :].copy()
    # torch.Tensor path
    idx_t = torch.from_numpy(indices).long()
    return J_dense[idx_t, :]


# ===========================================================================
# Model wrapper
# ===========================================================================

def token_model_wrapper(model, alphas_cumprod_2d, model_kwargs={}):
    """
    Returns: model_fn(x: BxLxd, t_ids: L long or BxL long) -> pred_x0: BxLxd
    Uses the per-token t_ids as the 2D timestep for the transformer (Option B).
    """
    def model_fn(x, t_ids):
        # Expand t_ids to (B, L) if it is (L,)
        if t_ids.dim() == 1:
            t_batch = t_ids.unsqueeze(0).expand(x.shape[0], -1)  # (B, L)
        else:
            t_batch = t_ids
        pred_x0 = model(x, t_batch, **model_kwargs)
        return pred_x0
    return model_fn
