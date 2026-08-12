# Token-Adaptive DPM-Solver: Implementation Plan

## Overview

This document is a self-contained implementation specification for a **token-adaptive DPM-Solver**
(`dpm_solver_token_adaptive.py` + `sample_seq2seq_adaptive_dpm.py`).

The core idea (from `inference_pipeline.md`) is to replace the single global timestep schedule
with a **per-token timestep matrix** `J (K x L)` derived from each token's empirical
loss-vs-logSNR curve. At every solver step `r` (out of K), token `i` is denoised from
`J[r,i]` toward `J[r+1,i]`, rather than every token moving through the same scalar `t`.

---

## 1. Context: What Exists and What Must Change

### 1.1 Existing adaptive-noise infrastructure (`diffuseq/gaussian_diffusion.py`)

| Symbol | Shape | Meaning |
|--------|-------|---------|
| `alphas_cumprod` | `(T, L)` | Per-token cumulative alpha at each timestep; T ~2000 |
| `sqrt_alphas_cumprod` | `(T, L)` | sqrt(alpha_bar_t^i) |
| `sqrt_one_minus_alphas_cumprod` | `(T, L)` | sqrt(1 - alpha_bar_t^i) |
| `_loss_history` | `(T//granu, L)` | Running sum of losses per (coarse-t, token-pos) |
| `_loss_history_count` | `(T//granu, L)` | Normalization counts |

Key method: `_load_time_schedule(path)` loads a saved `.npy` of shape `(T, L)` and calls
`update_time_discretized_parameters`.

### 1.2 Existing DPM-Solver (`dpm_solver_pytorch.py`)

`NoiseScheduleVP` is scalar: it holds a 1-D array `log_alpha_array` of shape `(1, T)` and
its methods (`marginal_lambda`, `marginal_alpha`, `marginal_std`, `inverse_lambda`) return scalars.

`DPM_Solver.sample()` currently:
1. Computes a global scalar timestep sequence `[t_T, ..., t_0]` of length K+1.
2. Runs multistep/singlestep updates where every token in `x (B x L x d)` moves through
   the **same** scalar `t`.

### 1.3 What the plan produces

Two new files:

| File | Role |
|------|------|
| `dpm_solver_token_adaptive.py` | New file containing `TokenAdaptiveDPM_Solver`, `build_token_timestep_matrix`, `token_model_wrapper` |
| `sample_seq2seq_adaptive_dpm.py` | Entry-point script: runs offline loss-profiling (steps 1-12) then adaptive sampling |

---

## 2. Mathematical Foundations

### 2.1 Token-wise noising (review)

The forward process for token `i` at timestep `t`:

    z_t^i = sqrt(alpha_bar_t^i) * z_0^i + sqrt(1 - alpha_bar_t^i) * eps^i

Token-wise log-SNR (half-logSNR following DPM-Solver convention):

    lambda_t^i = log(alpha_bar_t^i / (1 - alpha_bar_t^i)) / 2

Both `alpha_bar_t^i` and `lambda_t^i` vary across positions `i` because the schedule is adaptive.

### 2.2 Per-token timestep matrix construction (steps 1-12 of `inference_pipeline.md`)

**Offline profiling pass** (done once on a validation set of size B):

**Step 1 — Accumulate E[T, L]:**
For each `t in {T, ..., 1}` and each sample `b`:
- Sample `z_t^{b,i} = sqrt(alpha_bar_t^i) * z_0^{b,i} + sqrt(1-alpha_bar_t^i) * eps^{b,i}`
- Run `f_theta(z_t, h_x, t, lambda_t)` -> `z_hat_{0,b,t}`
- Compute `l_{b,t,i} = (1/d) || z_hat_{0,b,t}^i - z_0^{b,i} ||_2^2`

Average over the validation batch:
`E_{t,i} = (1/B) sum_b l_{b,t,i}`   shape: `(T, L)`

**Step 2 — Per-token loss curve:**
For each token `i`: `C_i = {(lambda_t^i, E_{t,i})}_{t=1}^T`

**Step 3 — Fit and differentiate:**
Fit a monotone smoothing spline (or Gaussian-smoothed finite differences) to `C_i`.
Compute slope `dE_i / d_lambda`.

**Step 4 — Node density (curvature proxy):**
`a_i(lambda) = -exp(lambda) * dE_i/d_lambda`   (>= 0; loss decreases as lambda increases)

**Step 5 — Cumulative distribution:**
`G_i(lambda) = integral_{lambda_min}^{lambda} a_i(u) du`
Normalized: `G_bar_i(lambda) = G_i(lambda) / G_i(lambda_max)`

**Step 6 — Select K nodes by inverting G_bar_i:**
`q_r = r / (K-1)`, r = 0, ..., K-1
`lambda_star_{r,i} = G_bar_i^{-1}(q_r)`

**Step 7 — Map to discrete timestep indices:**
`J_{r,i} = argmin_t |lambda_t^i - lambda_star_{r,i}|`

**Step 8 — Enforce denoising direction:**
Rows of J must be sorted so J[0, i] is largest t (noisiest) and J[K-1, i] is smallest t
(cleanest).

Result: `J in N^{K x L}` -- the per-token timestep matrix.

### 2.3 Token-heterogeneous DPM-Solver update

At solver step `r -> r+1`, token `i` transitions from `t_s^i = J[r, i]` to `t_t^i = J[r+1, i]`.

**DPM-Solver++ order-1 (DDIM style), token-wise:**

    h^i = lambda_{t_t}^i - lambda_{t_s}^i
    phi_1^i = expm1(-h^i)
    x_{t_t}^i = (sigma_{t_t}^i / sigma_{t_s}^i) * x_{t_s}^i
                - alpha_{t_t}^i * phi_1^i * x_hat_0^i

All token-wise coefficients are tensors of shape `(L,)`, broadcast to `(B, L, 1)`.

**Order-2 multistep** generalizes: correction term uses
`D1^i = (1/r0^i)(m_{prev,0}^i - m_{prev,1}^i)` where each `h`, `r0` is token-wise.

---

## 3. New Class: `TokenAdaptiveDPM_Solver`

### 3.1 Location
New file: `dpm_solver_token_adaptive.py`

### 3.2 Constructor signature

```python
class TokenAdaptiveDPM_Solver:
    def __init__(
        self,
        model_fn,             # callable: (x: BxLxd, t_ids: L) -> pred_x0: BxLxd
        alphas_cumprod_2d,    # torch.Tensor shape (T, L), on device
        algorithm_type="dpmsolver++",
        correcting_x0_fn=None,
    ):
        self.model_fn = model_fn
        self.alphas_cumprod_2d = alphas_cumprod_2d   # (T, L)
        self.algorithm_type = algorithm_type
        self.correcting_x0_fn = correcting_x0_fn
```

### 3.3 Internal helper: `_token_schedule_coeffs(t_ids)`

```
Input:  t_ids  shape (L,)   -- integer indices into the T-axis
Output: alpha_t  shape (L,)
        sigma_t  shape (L,)
        lambda_t shape (L,)  -- half-logSNR
```

```python
def _token_schedule_coeffs(self, t_ids):
    # t_ids: (L,) long tensor
    L = t_ids.shape[0]
    ac = self.alphas_cumprod_2d               # (T, L)
    ac_t = ac[t_ids, torch.arange(L, device=t_ids.device)]  # (L,) -- diagonal index
    alpha_t = torch.sqrt(ac_t)
    sigma_t = torch.sqrt(1.0 - ac_t)
    lambda_t = 0.5 * torch.log(ac_t / (1.0 - ac_t + 1e-9))  # half-logSNR
    return alpha_t, sigma_t, lambda_t
```

### 3.4 Token-wise first-order update

```python
def _token_first_update(self, x, t_s_ids, t_t_ids, model_s=None):
    """
    x          : (B, L, d)
    t_s_ids    : (L,)  source timestep per token
    t_t_ids    : (L,)  target timestep per token
    model_s    : (B, L, d) or None -- predicted x0 at source; computed if None
    """
    alpha_s, sigma_s, lambda_s = self._token_schedule_coeffs(t_s_ids)  # (L,)
    alpha_t, sigma_t, lambda_t = self._token_schedule_coeffs(t_t_ids)  # (L,)
    h     = lambda_t - lambda_s   # (L,)  h^i for each token
    phi_1 = torch.expm1(-h)       # (L,)  phi_1^i

    if model_s is None:
        model_s = self.model_fn(x, t_s_ids)   # (B, L, d)  -- pred_x0
        if self.correcting_x0_fn is not None:
            model_s = self.correcting_x0_fn(model_s)

    # Broadcast (L,) -> (1, L, 1) for operations on (B, L, d)
    def bcast(v): return v[None, :, None]

    x_t = (
        bcast(sigma_t / sigma_s) * x
        - bcast(alpha_t * phi_1) * model_s
    )
    return x_t, model_s
```

### 3.5 Token-wise second-order multistep update

```python
def _token_multistep_second_update(self, x, model_prev_list, t_prev_ids_list, t_t_ids):
    """
    Applies a 2nd-order Adams-style correction, analogous to
    DPM_Solver.multistep_dpm_solver_second_update but with per-token h values.
    """
    m0 = model_prev_list[-1]      # (B, L, d) -- most recent model eval
    m1 = model_prev_list[-2]      # (B, L, d) -- second most recent
    t0_ids = t_prev_ids_list[-1]  # (L,)
    t1_ids = t_prev_ids_list[-2]  # (L,)

    _, _,       lam_prev1   = self._token_schedule_coeffs(t1_ids)
    alpha_prev0, sigma_prev0, lam_prev0 = self._token_schedule_coeffs(t0_ids)
    alpha_t,    sigma_t,    lam_t       = self._token_schedule_coeffs(t_t_ids)

    h_0 = lam_prev0 - lam_prev1  # (L,) -- previous interval
    h   = lam_t     - lam_prev0  # (L,) -- current interval
    r0  = h_0 / h                 # (L,) -- ratio

    D1_0 = (1.0 / r0)[None, :, None] * (m0 - m1)  # (B, L, d)

    phi_1 = torch.expm1(-h)       # (L,)
    def bcast(v): return v[None, :, None]

    x_t = (
        bcast(sigma_t / sigma_prev0) * x
        - bcast(alpha_t * phi_1) * m0
        - 0.5 * bcast(alpha_t * phi_1) * D1_0
    )
    return x_t
```

### 3.6 Main sampling entry point

```python
def sample(
    self,
    x,                  # (B, L, d) -- noised input; token i is at t=J[0,i]
    J,                  # (K, L) long tensor -- per-token timestep matrix
    order=2,
    x_start=None,       # (B, L, d) -- source embeddings to anchor
    input_ids_mask=None,# (B, L, 1) -- 0=source, 1=target/pad
):
    """
    K-1 solver steps. Step r moves token i from J[r, i] to J[r+1, i].
    J must have rows in descending t order (noisiest first).
    """
    K = J.shape[0]  # number of timestep rows; K-1 intervals

    model_prev_list = []
    t_prev_ids_list = []

    for r in range(K - 1):
        t_s_ids = J[r]       # (L,)
        t_t_ids = J[r + 1]   # (L,)

        if r == 0 or order == 1 or len(model_prev_list) < 2:
            x, model_s = self._token_first_update(x, t_s_ids, t_t_ids)
        else:
            # Evaluate model at current state before updating
            model_s = self.model_fn(x, t_s_ids)
            if self.correcting_x0_fn is not None:
                model_s = self.correcting_x0_fn(model_s)
            x = self._token_multistep_second_update(
                x, model_prev_list, t_prev_ids_list, t_t_ids
            )

        # Update history (keep only last `order` entries)
        model_prev_list.append(model_s)
        t_prev_ids_list.append(t_s_ids)
        if len(model_prev_list) > order:
            model_prev_list.pop(0)
            t_prev_ids_list.pop(0)

        # Always anchor source tokens back to clean embeddings
        if input_ids_mask is not None and x_start is not None:
            x = torch.where(input_ids_mask == 0, x_start, x)

    return x
```

---

## 4. Offline Profiling Module: `build_token_timestep_matrix`

### 4.1 Function signature

```python
def build_token_timestep_matrix(
    model,              # frozen f_theta: callable
    diffusion,          # GaussianDiffusion with 2-D alphas_cumprod (T, L)
    data_loader,        # iterable yielding (z0, cond) batches
    K,                  # number of solver nodes to select
    device,
    T_subset=None,      # if int, randomly sub-sample T timesteps from [0, T)
    smooth_sigma=5.0,   # sigma for Gaussian smoothing of E_{t,i} curves
    profile_batches=10, # number of batches to process
) -> np.ndarray:        # J: shape (K, L), dtype int64
```

### 4.2 Step-by-step implementation

```python
@torch.no_grad()
def build_token_timestep_matrix(model, diffusion, data_loader, K, device,
                                 T_subset=None, smooth_sigma=5.0,
                                 profile_batches=10):
    from scipy.ndimage import gaussian_filter1d
    from scipy.integrate import cumulative_trapezoid

    T = diffusion.num_timesteps
    L = diffusion.token_max_length
    ac = diffusion.alphas_cumprod  # (T, L) numpy

    # ---- Step 1: accumulate E[T, L] ----
    t_range = np.arange(T) if T_subset is None else np.random.choice(T, T_subset, replace=False)
    E_sum   = np.zeros((T, L), dtype=np.float64)
    E_count = np.zeros((T, L), dtype=np.float64)

    for batch_idx, (z0, cond) in enumerate(data_loader):
        if batch_idx >= profile_batches:
            break
        z0 = z0.to(device)                          # (B, L, d)
        B = z0.shape[0]
        for t_int in t_range:
            t_tensor = torch.full((B,), t_int, device=device, dtype=torch.long)
            z_t = diffusion.q_sample(z0, t_tensor)  # (B, L, d)
            pred_x0 = model(z_t, t_tensor, **cond)  # (B, L, d)
            loss = ((pred_x0 - z0) ** 2).mean(-1)   # (B, L) -- mean over hidden dim
            E_sum[t_int]   += loss.sum(0).cpu().numpy()
            E_count[t_int] += B

    # Avoid divide-by-zero for unvisited timesteps
    E_count = np.where(E_count == 0, 1.0, E_count)
    E = E_sum / E_count                              # (T, L)

    # ---- Steps 2-7: per-token schedule ----
    J = np.zeros((K, L), dtype=np.int64)
    lambda_array = 0.5 * np.log(ac / (1.0 - ac + 1e-9))  # (T, L) half-logSNR

    for i in range(L):
        lambda_i = lambda_array[:, i]   # (T,) -- sorted ascending (more noise = lower lambda)
        E_i      = E[:, i]              # (T,) -- loss at each t for token i

        # Step 3: smooth
        if smooth_sigma > 0:
            E_i_smooth = gaussian_filter1d(E_i, sigma=smooth_sigma)
        else:
            E_i_smooth = E_i.copy()

        # Step 4: slope via finite differences (T-1 midpoints)
        d_lam  = np.diff(lambda_i)             # (T-1,)
        d_E    = np.diff(E_i_smooth)           # (T-1,)
        # Avoid division by zero
        d_lam_safe = np.where(np.abs(d_lam) < 1e-9, 1e-9, d_lam)
        slope  = d_E / d_lam_safe              # dE/dlambda at midpoints

        lam_mid = (lambda_i[:-1] + lambda_i[1:]) / 2.0  # (T-1,)

        # Step 5: node density a_i(lambda) = -exp(lambda) * dE/dlambda
        a_i = -np.exp(lam_mid) * slope
        a_i = np.clip(a_i, 0, None)            # ensure non-negative

        # Step 6: cumulative trapezoid
        G_i = cumulative_trapezoid(a_i, lam_mid, initial=0.0)  # (T-1,)
        G_max = G_i[-1]

        if G_max < 1e-9:
            # Degenerate: uniform logSNR spacing (fallback)
            lam_star_i = np.linspace(lam_mid[0], lam_mid[-1], K)
        else:
            G_norm = G_i / G_max                    # normalize to [0, 1]
            q      = np.linspace(0.0, 1.0, K)
            # Invert via linear interpolation
            lam_star_i = np.interp(q, G_norm, lam_mid)  # (K,)

        # Step 7: map lambda* -> discrete t index
        for r in range(K):
            J[r, i] = np.argmin(np.abs(lambda_i - lam_star_i[r]))

    # Step 8: enforce descending t order (noisiest first)
    # lambda increases as t decreases (less noise), so argmin of lambda = largest t
    # After inversion, lam_star_i[0] should be smallest lambda (noisiest).
    # Sort each column of J in descending order of t to be safe.
    J = np.sort(J, axis=0)[::-1].copy()  # (K, L): row 0 = max t, row K-1 = min t

    return J  # shape (K, L)
```

### 4.3 Caching

```python
# Save
np.save("J_adaptive_K20.npy", J)

# Load
J = np.load("J_adaptive_K20.npy")
```

---

## 5. Token-wise Model Wrapper

The model takes a scalar t for time conditioning. Two options:

### Option A — Global mean-t (no model change, recommended for prototyping)

```python
def token_model_wrapper(model, alphas_cumprod_2d, model_kwargs={}):
    """
    Returns: model_fn(x: BxLxd, t_ids: L long) -> pred_x0: BxLxd
    Uses the column-mean of t_ids as the scalar timestep for the transformer.
    """
    def model_fn(x, t_ids):
        # t_ids: (L,) long
        t_scalar = t_ids.float().mean().round().long()
        # Expand to batch size
        t_batch  = t_scalar.expand(x.shape[0])  # (B,)
        pred_x0  = model(x, t_batch, **model_kwargs)
        return pred_x0
    return model_fn
```

### Option B — Per-token time conditioning (requires `transformer_model.py` edit)

Modify `transformer_model.py`:
1. Change signature to accept `t: (B, L)` or `t: (L,)`.
2. Compute `time_emb = self.time_embed(t)` element-wise, shape `(B, L, d_time)`.
3. Add `time_emb` to per-token hidden states at the input projection stage.

This is the preferred approach for full accuracy but should follow after Option A is validated.

---

## 6. Entry-Point Script: `sample_seq2seq_adaptive_dpm.py`

### 6.1 Additional CLI arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--K` | int | 20 | Number of solver nodes (= denoising steps + 1) |
| `--J_path` | str | `""` | Path to pre-computed `J.npy`. If empty, runs profiling. |
| `--profile_batches` | int | 10 | Number of validation batches for profiling |
| `--smooth_sigma` | float | 5.0 | Gaussian smoothing sigma for loss curves |
| `--T_subset` | int | 0 | Sub-sample this many timesteps for profiling (0 = all T) |
| `--solver_order` | int | 2 | DPM-Solver order (1 or 2) |

### 6.2 High-level main() flow

```python
def main():
    # --- 1. Parse & load (identical to sample_seq2seq_dpmSolver.py) ---
    args   = create_argparser().parse_args()
    model, diffusion = create_model_and_diffusion(...)
    model.load_state_dict(...)
    diffusion._load_time_schedule(args.time_schedule_path)
    model.eval()

    # --- 2. Build or load J ---
    if args.J_path and os.path.exists(args.J_path):
        J = np.load(args.J_path)           # (K, L)
        print(f"Loaded J from {args.J_path}")
    else:
        J = build_token_timestep_matrix(
            model=model, diffusion=diffusion,
            data_loader=data_valid_profiler,   # a separate loader for profiling
            K=args.K, device=device,
            T_subset=args.T_subset or None,
            smooth_sigma=args.smooth_sigma,
            profile_batches=args.profile_batches,
        )
        save_path = args.J_path or "J_adaptive.npy"
        np.save(save_path, J)
        print(f"Saved J to {save_path}")

    J_tensor = torch.from_numpy(J).long().to(device)   # (K, L)

    # --- 3. Build model_fn and solver ---
    alphas_cumprod_t = torch.from_numpy(diffusion.alphas_cumprod).float().to(device)
    model_fn = token_model_wrapper(model, alphas_cumprod_t, model_kwargs={})

    def rounding_corrector(x0):
        rounded, _ = denoised_fn_round(args, model_emb, x0)
        return rounded

    solver = TokenAdaptiveDPM_Solver(
        model_fn=model_fn,
        alphas_cumprod_2d=alphas_cumprod_t,
        algorithm_type="dpmsolver++",
        correcting_x0_fn=rounding_corrector,
    )

    # --- 4. Sampling loop ---
    for cond in data_valid:
        input_ids_x  = cond.pop('input_ids').to(device)
        x_start      = model.get_embeds(input_ids_x)
        input_ids_mask = cond.pop('input_mask')
        noise        = torch.randn_like(x_start)
        input_ids_mask_3d = input_ids_mask.unsqueeze(-1).expand_as(x_start).to(device)

        # Start from the noisiest timestep per token: x at J[0, i]
        # The J matrix's first row already gives the noisiest t per token.
        # Use a batch-level q_sample that respects per-token t:
        x_noised = sample_x_at_J0(x_start, J_tensor[0], diffusion, noise, device)
        # (For simplicity in prototype: use global max-t noising)
        # x_noised = torch.where(input_ids_mask_3d == 0, x_start, noise)

        x_sample = solver.sample(
            x_noised,
            J=J_tensor,
            order=args.solver_order,
            x_start=x_start,
            input_ids_mask=input_ids_mask_3d,
        )

        # Decode and write output (identical to sample_seq2seq_dpmSolver.py)
        decode_and_save(x_sample, ...)
```

### 6.3 Helper: `sample_x_at_J0`

```python
def sample_x_at_J0(x_start, t_ids_row0, diffusion, noise, device):
    """
    Noise x_start to the per-token starting timesteps in J[0].
    t_ids_row0 : (L,) long tensor -- per-token starting t
    x_start    : (B, L, d)
    Returns    : (B, L, d)
    """
    ac = torch.from_numpy(diffusion.alphas_cumprod).float().to(device)  # (T, L)
    L  = x_start.shape[1]
    # ac_t[i] = alpha_bar_{J[0,i]}^i
    ac_t  = ac[t_ids_row0, torch.arange(L, device=device)]  # (L,)
    alpha = torch.sqrt(ac_t)[None, :, None]    # (1, L, 1)
    sigma = torch.sqrt(1.0 - ac_t)[None, :, None]
    return alpha * x_start + sigma * noise
```

---

## 7. File Map and Dependencies

```
diffuseqv2_tgmdlm/
|-- dpm_solver_pytorch.py               [EXISTING -- do not modify]
|-- dpm_solver_token_adaptive.py        [NEW -- main deliverable]
|       |-- TokenAdaptiveDPM_Solver
|       |-- build_token_timestep_matrix
|       `-- token_model_wrapper
|-- sample_seq2seq_dpmSolver.py         [EXISTING -- do not modify]
|-- sample_seq2seq_adaptive_dpm.py      [NEW -- entry-point script]
|-- token_adaptive_dpm_solver_plan.md   [THIS FILE]
|-- diffuseq/
|   |-- gaussian_diffusion.py           [EXISTING -- provides 2-D schedule]
|   `-- transformer_model.py            [EXISTING -- Option B: per-token t]
`-- J_adaptive_K<K>.npy                [GENERATED -- cached J matrix]
```

---

## 8. Implementation Order

| Step | Task | Files |
|------|------|-------|
| P1 | Create `dpm_solver_token_adaptive.py` with imports and stubs | new |
| P2 | Implement `_token_schedule_coeffs` + unit test vs scalar schedule | token_adaptive |
| P3 | Implement `_token_first_update` (order-1) | token_adaptive |
| P4 | Implement `TokenAdaptiveDPM_Solver.sample` with order=1 only | token_adaptive |
| P5 | Regression test: uniform J vs original DPM-Solver output | both |
| P6 | Implement `_token_multistep_second_update` + extend sample to order=2 | token_adaptive |
| P7 | Implement `build_token_timestep_matrix` profiling function | token_adaptive |
| P8 | Implement `token_model_wrapper` (Option A) | token_adaptive |
| P9 | Implement `sample_x_at_J0` helper | new script |
| P10 | Create `sample_seq2seq_adaptive_dpm.py` with full CLI + loop | new script |
| P11 | End-to-end test: profile on 5 batches, sample, compute BLEU | both |
| P12 (opt) | Option B: per-token t conditioning in transformer_model.py | transformer_model |

---

## 9. Testing Checkpoints

### 9.1 Unit test: `_token_schedule_coeffs`
Construct a 2-D `alphas_cumprod` where all columns are identical (= 1-D schedule).
Verify that `alpha_t`, `sigma_t`, `lambda_t` match `NoiseScheduleVP.marginal_alpha/std/lambda`
for the same timestep index.

### 9.2 Regression test: uniform J
Set `J[r, i] = global_timesteps[r]` for all `i` (same scalar schedule for every token).
Run `TokenAdaptiveDPM_Solver.sample` and original `DPM_Solver.sample` on the same noised
input. Verify `||output_adaptive - output_original||_2 < 1e-4` (up to float precision).

### 9.3 Profiling sanity: monotone loss
After computing `E_{t,i}`, verify that Gaussian-smoothed `E_i` is monotonically decreasing
in `lambda` for > 95% of token positions. Tokens that fail indicate noisy loss estimates;
increase `smooth_sigma` or `profile_batches`.

### 9.4 Schedule visual inspection
Plot `J[:, i]` for a few token positions `i`. Expect:
- Row 0 (noisiest): J[0, i] close to T for all i
- Row K-1 (cleanest): J[K-1, i] = small value, varying across tokens
- Tokens with steep loss curves (hard tokens) should have denser node placement at high-t

### 9.5 End-to-end quality
Run `sample_seq2seq_adaptive_dpm.py` with K=20 and compare BLEU/RougeL against
`sample_seq2seq_dpmSolver.py` with steps=20.
The adaptive schedule should match or improve quality.

---

## 10. Key Invariants to Preserve

1. **Source tokens are never denoised.** The mask `input_ids_mask == 0` is re-applied after
   every solver step (identical to the original `DPM_Solver.sample`).

2. **The profiling pass is `@torch.no_grad()`.** No gradients stored during profiling.

3. **J rows are sorted descending in t** (largest t first). Corresponds to noisiest -> cleanest.

4. **Degenerate tokens** (flat loss curve, G_max ~ 0) fall back to uniform logSNR spacing
   across [lambda_min^i, lambda_max^i].

5. **Float precision**: `alphas_cumprod_2d` must be `float32` on GPU; avoid `float64` in the
   inner solver loop to maintain speed parity with the original solver.

6. **Model must stay frozen** during both profiling and sampling.

---

## 11. Minimal Viable Prototype Checklist

- [ ] `dpm_solver_token_adaptive.py` created with all three components
- [ ] `TokenAdaptiveDPM_Solver.sample` runs without error on a dummy `(B=2, L=128, d=128)` input
- [ ] `build_token_timestep_matrix` runs on 1 batch and saves `J.npy` of correct shape `(K, L)`
- [ ] `sample_seq2seq_adaptive_dpm.py` completes one full iteration on validation split
- [ ] Regression test (uniform J) passes within `1e-4` L2 of original solver output
- [ ] BLEU >= baseline DPM-Solver at K=20 steps

