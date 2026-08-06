"""Modular validation-trajectory loss–log-SNR scheduler.

Implements the simple along-path approximation:
1) frozen checkpoint, 2) validation loss curves E_i(lambda),
3) smoothing and dE_i/dlambda, 4) a_i=-exp(lambda)dE_i/dlambda,
5) equal sqrt(a_i)-mass node selection, 6) K x L reduced schedule.

Adapt the four small adapter classes to your codebase.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Protocol, Sequence
import json
import warnings

import numpy as np
import torch
from torch import Tensor, nn
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter
from sklearn.isotonic import IsotonicRegression


@dataclass(frozen=True)
class CalibrationConfig:
    num_dense_steps: int = 2000
    max_batches: Optional[int] = None
    reuse_noise_across_timesteps: bool = True
    latent_loss_reduction: str = "mean"  # mean | sum
    device: str = "cuda"
    dtype: torch.dtype = torch.float32


@dataclass(frozen=True)
class CurveFitConfig:
    method: str = "isotonic_pchip"  # isotonic_pchip | savgol | pchip_raw
    dense_grid_size: int = 4096
    savgol_window: int = 51
    savgol_polyorder: int = 3
    monitor_floor: float = 1e-8
    positive_slope_policy: str = "clip"  # clip | warn | keep
    min_valid_points: int = 8


@dataclass(frozen=True)
class NodeSelectionConfig:
    num_nodes: int = 10
    force_endpoints: bool = True
    strict_monotonic_indices: bool = True


@dataclass
class CalibrationResult:
    loss_sum: Tensor                # [T,L], CPU float64
    valid_count: Tensor             # [T,L], CPU float64
    mean_loss: Tensor               # [T,L], CPU float64
    lambda_schedule: Tensor         # [T,L], CPU float64
    alpha_bar_schedule: Tensor      # [T,L], CPU float64


@dataclass
class PositionCurve:
    position: int
    lambda_raw: np.ndarray
    error_raw: np.ndarray
    lambda_dense: np.ndarray
    error_smooth: np.ndarray
    derivative: np.ndarray
    monitor: np.ndarray
    density: np.ndarray
    cumulative_mass: np.ndarray
    monotonicity_violation_fraction: float
    total_monitor_mass: float


@dataclass
class ReducedSchedule:
    indices: np.ndarray             # [K,L]
    lambda_values: np.ndarray       # [K,L]
    alpha_bar_values: np.ndarray    # [K,L]
    quantiles: np.ndarray           # [K]


# ---------------------------------------------------------------------
# Adapters: customize these for your repository
# ---------------------------------------------------------------------

class DiffusionModelAdapter(Protocol):
    def predict_z0(
        self,
        *,
        z_t: Tensor,
        source_cond: Any,
        timestep: int,
        lambda_vec: Tensor,
    ) -> Tensor:
        """Return predicted clean latent, shape [B,L,D]."""
        ...


class TargetLatentAdapter(Protocol):
    def encode_target(self, batch: Mapping[str, Any]) -> Tensor:
        ...


class SourceConditioningAdapter(Protocol):
    def encode_source(self, batch: Mapping[str, Any]) -> Any:
        ...


class ValidMaskAdapter(Protocol):
    def get_valid_mask(self, batch: Mapping[str, Any], z0: Tensor) -> Tensor:
        ...


class ReverseStepAdapter(Protocol):
    def step(
        self,
        *,
        z_current: Tensor,
        z0_hat: Tensor,
        alpha_bar_current: Tensor,
        alpha_bar_next: Tensor,
        stochastic: bool,
    ) -> Tensor:
        ...


# ---------------------------------------------------------------------
# Schedule utilities
# ---------------------------------------------------------------------

def alpha_bar_to_log_snr(alpha_bar: Tensor, eps: float = 1e-12) -> Tensor:
    alpha_bar = alpha_bar.clamp(min=eps, max=1.0 - eps)
    return torch.log(alpha_bar) - torch.log1p(-alpha_bar)


def validate_dense_schedule(alpha_bar_schedule: Tensor) -> None:
    if alpha_bar_schedule.ndim != 2:
        raise ValueError("alpha_bar_schedule must have shape [T,L]")
    if not torch.isfinite(alpha_bar_schedule).all():
        raise ValueError("Schedule contains non-finite values")
    if (alpha_bar_schedule <= 0).any() or (alpha_bar_schedule >= 1).any():
        raise ValueError("All alpha_bar values must lie in (0,1)")


# ---------------------------------------------------------------------
# Stage 1: estimate E_i(lambda_t^i) on validation data
# ---------------------------------------------------------------------

class ValidationTrajectoryCalibrator:
    def __init__(
        self,
        *,
        model: DiffusionModelAdapter,
        target_adapter: TargetLatentAdapter,
        source_adapter: SourceConditioningAdapter,
        mask_adapter: ValidMaskAdapter,
        alpha_bar_schedule: Tensor,
        config: CalibrationConfig,
    ) -> None:
        validate_dense_schedule(alpha_bar_schedule)
        if alpha_bar_schedule.shape[0] != config.num_dense_steps:
            raise ValueError("Schedule T does not match config.num_dense_steps")
        self.model = model
        self.target_adapter = target_adapter
        self.source_adapter = source_adapter
        self.mask_adapter = mask_adapter
        self.alpha_bar_schedule = alpha_bar_schedule.detach().cpu().double()
        self.lambda_schedule = alpha_bar_to_log_snr(self.alpha_bar_schedule).double()
        self.config = config

    @torch.inference_mode()
    def run(self, validation_loader: Iterable[Mapping[str, Any]]) -> CalibrationResult:
        device = torch.device(self.config.device)
        schedule = self.alpha_bar_schedule.to(device=device, dtype=self.config.dtype)
        T, L = schedule.shape

        loss_sum = torch.zeros(T, L, dtype=torch.float64)
        valid_count = torch.zeros(T, L, dtype=torch.float64)

        for batch_idx, batch in enumerate(validation_loader):
            if self.config.max_batches is not None and batch_idx >= self.config.max_batches:
                break

            z0 = self.target_adapter.encode_target(batch).to(device=device, dtype=self.config.dtype)
            if z0.ndim != 3:
                raise ValueError(f"Expected z0 [B,L,D], got {tuple(z0.shape)}")
            B, batch_L, _ = z0.shape
            if batch_L != L:
                raise ValueError(f"Batch L={batch_L}, schedule L={L}")

            source_cond = _move_nested_to_device(self.source_adapter.encode_source(batch), device)
            valid_mask = self.mask_adapter.get_valid_mask(batch, z0).to(device=device, dtype=self.config.dtype)
            if valid_mask.shape != (B, L):
                raise ValueError(f"Expected mask {(B,L)}, got {tuple(valid_mask.shape)}")

            common_noise = torch.randn_like(z0) if self.config.reuse_noise_across_timesteps else None

            for t in range(T):
                alpha_t = schedule[t].unsqueeze(0).expand(B, -1)        # [B,L]
                lambda_t = alpha_bar_to_log_snr(alpha_t)               # [B,L]
                eps = common_noise if common_noise is not None else torch.randn_like(z0)

                z_t = (
                    alpha_t.sqrt().unsqueeze(-1) * z0
                    + (1.0 - alpha_t).sqrt().unsqueeze(-1) * eps
                )

                z0_hat = self.model.predict_z0(
                    z_t=z_t,
                    source_cond=source_cond,
                    timestep=t,
                    lambda_vec=lambda_t,
                )
                if z0_hat.shape != z0.shape:
                    raise ValueError("Model output shape differs from z0")

                sq = (z0_hat - z0).pow(2)
                if self.config.latent_loss_reduction == "mean":
                    per_token = sq.mean(dim=-1)                         # [B,L]
                elif self.config.latent_loss_reduction == "sum":
                    per_token = sq.sum(dim=-1)
                else:
                    raise ValueError("latent_loss_reduction must be mean or sum")

                loss_sum[t] += (per_token * valid_mask).sum(dim=0).double().cpu()
                valid_count[t] += valid_mask.sum(dim=0).double().cpu()

        mean_loss = loss_sum / valid_count.clamp_min(1.0)
        mean_loss[valid_count == 0] = torch.nan

        return CalibrationResult(
            loss_sum=loss_sum,
            valid_count=valid_count,
            mean_loss=mean_loss,
            lambda_schedule=self.lambda_schedule.clone(),
            alpha_bar_schedule=self.alpha_bar_schedule.clone(),
        )


# ---------------------------------------------------------------------
# Stage 2: smooth, differentiate, build a_i and sqrt(a_i)
# ---------------------------------------------------------------------

class RecoveryCurveEstimator:
    def __init__(self, config: CurveFitConfig) -> None:
        self.config = config

    def fit_all(self, calibration: CalibrationResult) -> list[PositionCurve]:
        _, L = calibration.mean_loss.shape
        return [
            self.fit_position(
                position=i,
                lambda_values=calibration.lambda_schedule[:, i].numpy(),
                error_values=calibration.mean_loss[:, i].numpy(),
            )
            for i in range(L)
        ]

    def fit_position(
        self,
        *,
        position: int,
        lambda_values: np.ndarray,
        error_values: np.ndarray,
    ) -> PositionCurve:
        valid = np.isfinite(lambda_values) & np.isfinite(error_values)
        x = np.asarray(lambda_values[valid], dtype=np.float64)
        y = np.asarray(error_values[valid], dtype=np.float64)
        if x.size < self.config.min_valid_points:
            raise ValueError(f"Position {position}: too few valid points")

        order = np.argsort(x)
        x, y = _merge_duplicate_x(x[order], y[order])
        raw_slopes = np.diff(y) / np.diff(x)
        violation_fraction = float(np.mean(raw_slopes > 0)) if raw_slopes.size else 0.0

        x_dense = np.linspace(x[0], x[-1], self.config.dense_grid_size)
        y_dense = self._smooth(x, y, x_dense)
        derivative = np.gradient(y_dense, x_dense)

        if self.config.positive_slope_policy == "clip":
            derivative = np.minimum(derivative, 0.0)
        elif self.config.positive_slope_policy == "warn" and np.any(derivative > 0):
            warnings.warn(f"Position {position}: positive slopes remain")
        elif self.config.positive_slope_policy not in {"clip", "warn", "keep"}:
            raise ValueError("Unknown positive_slope_policy")

        monitor = np.maximum(-np.exp(x_dense) * derivative, self.config.monitor_floor)
        density = np.sqrt(monitor)
        cumulative = cumulative_trapezoid(density, x_dense, initial=0.0)
        total_mass = float(cumulative[-1])
        if not np.isfinite(total_mass) or total_mass <= 0:
            raise ValueError(f"Position {position}: invalid monitor mass")
        cumulative /= total_mass

        return PositionCurve(
            position=position,
            lambda_raw=x,
            error_raw=y,
            lambda_dense=x_dense,
            error_smooth=y_dense,
            derivative=derivative,
            monitor=monitor,
            density=density,
            cumulative_mass=cumulative,
            monotonicity_violation_fraction=violation_fraction,
            total_monitor_mass=total_mass,
        )

    def _smooth(self, x: np.ndarray, y: np.ndarray, x_dense: np.ndarray) -> np.ndarray:
        if self.config.method == "isotonic_pchip":
            iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
            y_fit = iso.fit_transform(x, y)
        elif self.config.method == "savgol":
            window = _safe_savgol_window(
                self.config.savgol_window, len(y), self.config.savgol_polyorder
            )
            y_fit = savgol_filter(y, window, self.config.savgol_polyorder, mode="interp")
        elif self.config.method == "pchip_raw":
            y_fit = y
        else:
            raise ValueError("Unknown smoothing method")

        return np.asarray(PchipInterpolator(x, y_fit, extrapolate=False)(x_dense))


# ---------------------------------------------------------------------
# Stage 3: lambda* = G_i^{-1}(r/(K-1)) and map to dense indices
# ---------------------------------------------------------------------

class EqualMassNodeSelector:
    """Select K nodes with equal cumulative sqrt(a_i)-mass."""

    def __init__(self, config: NodeSelectionConfig) -> None:
        if config.num_nodes < 2:
            raise ValueError("num_nodes must be >= 2")
        self.config = config

    def select(
        self,
        *,
        curves: Sequence[PositionCurve],
        lambda_schedule: Tensor,
        alpha_bar_schedule: Tensor,
    ) -> ReducedSchedule:
        lam = lambda_schedule.cpu().numpy()
        alpha = alpha_bar_schedule.cpu().numpy()
        T, L = lam.shape
        if len(curves) != L:
            raise ValueError("Number of curves must equal sequence length L")

        K = self.config.num_nodes
        quantiles = np.linspace(0.0, 1.0, K)
        indices = np.empty((K, L), dtype=np.int64)
        lambda_values = np.empty((K, L), dtype=np.float64)
        alpha_values = np.empty((K, L), dtype=np.float64)

        for i, curve in enumerate(curves):
            # Exact numerical implementation of lambda*_{r,i}=G_i^{-1}(r/(K-1)).
            lambda_star = np.interp(
                quantiles,
                curve.cumulative_mass,
                curve.lambda_dense,
            )

            dense_lambda_i = lam[:, i]
            idx = np.array([
                int(np.argmin(np.abs(dense_lambda_i - value)))
                for value in lambda_star
            ])

            if self.config.force_endpoints:
                idx[0] = int(np.argmin(dense_lambda_i))
                idx[-1] = int(np.argmax(dense_lambda_i))

            if self.config.strict_monotonic_indices:
                idx = _repair_in_lambda_order(idx, dense_lambda_i)

            indices[:, i] = idx
            lambda_values[:, i] = dense_lambda_i[idx]
            alpha_values[:, i] = alpha[idx, i]

        return ReducedSchedule(
            indices=indices,
            lambda_values=lambda_values,
            alpha_bar_values=alpha_values,
            quantiles=quantiles,
        )


# ---------------------------------------------------------------------
# Stage 4: K-call reduced sampler
# ---------------------------------------------------------------------

class TokenWiseReducedSampler:
    def __init__(
        self,
        *,
        model: DiffusionModelAdapter,
        reverse_step: ReverseStepAdapter,
        reduced_schedule: ReducedSchedule,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.model = model
        self.reverse_step = reverse_step
        self.schedule = reduced_schedule
        self.device = torch.device(device)
        self.dtype = dtype

    @torch.inference_mode()
    def sample(self, *, initial_latent: Tensor, source_cond: Any, stochastic: bool = True) -> Tensor:
        z = initial_latent.to(self.device, self.dtype)
        source_cond = _move_nested_to_device(source_cond, self.device)
        B, L, _ = z.shape

        lambda_values = torch.as_tensor(
            self.schedule.lambda_values, device=self.device, dtype=self.dtype
        )
        alpha_values = torch.as_tensor(
            self.schedule.alpha_bar_values, device=self.device, dtype=self.dtype
        )
        K, schedule_L = lambda_values.shape
        if schedule_L != L:
            raise ValueError("Schedule L differs from latent L")

        for r in range(K - 1):
            lambda_r = lambda_values[r].unsqueeze(0).expand(B, -1)
            alpha_current = alpha_values[r].unsqueeze(0).expand(B, -1)
            alpha_next = alpha_values[r + 1].unsqueeze(0).expand(B, -1)

            z0_hat = self.model.predict_z0(
                z_t=z,
                source_cond=source_cond,
                timestep=r,
                lambda_vec=lambda_r,
            )
            z = self.reverse_step.step(
                z_current=z,
                z0_hat=z0_hat,
                alpha_bar_current=alpha_current,
                alpha_bar_next=alpha_next,
                stochastic=stochastic,
            )

        final_lambda = lambda_values[-1].unsqueeze(0).expand(B, -1)
        return self.model.predict_z0(
            z_t=z,
            source_cond=source_cond,
            timestep=K - 1,
            lambda_vec=final_lambda,
        )


# ---------------------------------------------------------------------
# Generic reference adapters
# ---------------------------------------------------------------------

class GenericTorchModelAdapter:
    """Edit predict_z0 to match your model's exact call signature."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def predict_z0(self, *, z_t: Tensor, source_cond: Any, timestep: int, lambda_vec: Tensor) -> Tensor:
        t = torch.full((z_t.shape[0],), timestep, dtype=torch.long, device=z_t.device)
        output = self.model(
            z_t=z_t,
            encoder_hidden_states=source_cond,
            timesteps=t,
            log_snr=lambda_vec,
        )
        if isinstance(output, Tensor):
            return output
        if hasattr(output, "sample"):
            return output.sample
        if isinstance(output, Mapping) and "z0_hat" in output:
            return output["z0_hat"]
        raise TypeError("Modify GenericTorchModelAdapter to extract z0_hat")


class BatchTargetAdapter:
    def __init__(self, key: str = "z0") -> None:
        self.key = key
    def encode_target(self, batch: Mapping[str, Any]) -> Tensor:
        return batch[self.key]


class BatchSourceAdapter:
    def __init__(self, key: str = "source_cond") -> None:
        self.key = key
    def encode_source(self, batch: Mapping[str, Any]) -> Any:
        return batch[self.key]


class BatchMaskAdapter:
    def __init__(self, key: str = "attention_mask") -> None:
        self.key = key
    def get_valid_mask(self, batch: Mapping[str, Any], z0: Tensor) -> Tensor:
        return batch.get(self.key, torch.ones(z0.shape[:2], device=z0.device))


class GaussianPosteriorReverseStep:
    """Reference x0-prediction coarse posterior update; verify against your codebase."""
    def __init__(self, eps: float = 1e-12) -> None:
        self.eps = eps

    def step(
        self,
        *,
        z_current: Tensor,
        z0_hat: Tensor,
        alpha_bar_current: Tensor,
        alpha_bar_next: Tensor,
        stochastic: bool,
    ) -> Tensor:
        a_c = alpha_bar_current.clamp(self.eps, 1.0 - self.eps)
        a_n = alpha_bar_next.clamp(self.eps, 1.0 - self.eps)
        if not torch.all(a_n >= a_c):
            raise ValueError("Expected alpha_bar_next >= alpha_bar_current")

        alpha_coarse = (a_c / a_n).clamp(self.eps, 1.0)
        beta_coarse = 1.0 - alpha_coarse
        variance = ((1.0 - a_n) / (1.0 - a_c) * beta_coarse).clamp_min(self.eps)
        c = torch.sqrt(a_n) * beta_coarse / (1.0 - a_c)
        d = torch.sqrt(alpha_coarse) * (1.0 - a_n) / (1.0 - a_c)
        mean = c.unsqueeze(-1) * z0_hat + d.unsqueeze(-1) * z_current
        return mean if not stochastic else mean + variance.sqrt().unsqueeze(-1) * torch.randn_like(z_current)


# ---------------------------------------------------------------------
# Convenience API and persistence
# ---------------------------------------------------------------------

def build_reduced_schedule(
    *,
    model: DiffusionModelAdapter,
    validation_loader: Iterable[Mapping[str, Any]],
    target_adapter: TargetLatentAdapter,
    source_adapter: SourceConditioningAdapter,
    mask_adapter: ValidMaskAdapter,
    alpha_bar_schedule: Tensor,
    calibration_config: CalibrationConfig,
    curve_config: CurveFitConfig,
    node_config: NodeSelectionConfig,
) -> tuple[CalibrationResult, list[PositionCurve], ReducedSchedule]:
    calibration = ValidationTrajectoryCalibrator(
        model=model,
        target_adapter=target_adapter,
        source_adapter=source_adapter,
        mask_adapter=mask_adapter,
        alpha_bar_schedule=alpha_bar_schedule,
        config=calibration_config,
    ).run(validation_loader)

    curves = RecoveryCurveEstimator(curve_config).fit_all(calibration)
    schedule = EqualMassNodeSelector(node_config).select(
        curves=curves,
        lambda_schedule=calibration.lambda_schedule,
        alpha_bar_schedule=calibration.alpha_bar_schedule,
    )
    return calibration, curves, schedule


def save_outputs(
    *,
    calibration: CalibrationResult,
    curves: Sequence[PositionCurve],
    schedule: ReducedSchedule,
    output_dir: str | Path,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "calibration.npz",
        loss_sum=calibration.loss_sum.numpy(),
        valid_count=calibration.valid_count.numpy(),
        mean_loss=calibration.mean_loss.numpy(),
        lambda_schedule=calibration.lambda_schedule.numpy(),
        alpha_bar_schedule=calibration.alpha_bar_schedule.numpy(),
    )
    np.savez_compressed(
        out / "reduced_schedule.npz",
        indices=schedule.indices,
        lambda_values=schedule.lambda_values,
        alpha_bar_values=schedule.alpha_bar_values,
        quantiles=schedule.quantiles,
    )
    summary = []
    curve_dir = out / "curves"
    curve_dir.mkdir(exist_ok=True)
    for c in curves:
        np.savez_compressed(
            curve_dir / f"position_{c.position:04d}.npz",
            lambda_raw=c.lambda_raw,
            error_raw=c.error_raw,
            lambda_dense=c.lambda_dense,
            error_smooth=c.error_smooth,
            derivative=c.derivative,
            monitor=c.monitor,
            density=c.density,
            cumulative_mass=c.cumulative_mass,
        )
        summary.append({
            "position": c.position,
            "monotonicity_violation_fraction": c.monotonicity_violation_fraction,
            "total_monitor_mass": c.total_monitor_mass,
        })
    (out / "curve_summary.json").write_text(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------

def _move_nested_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {k: _move_nested_to_device(v, device) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_nested_to_device(v, device) for v in value)
    if isinstance(value, list):
        return [_move_nested_to_device(v, device) for v in value]
    return value


def _merge_duplicate_x(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique_x, inverse = np.unique(x, return_inverse=True)
    sums = np.zeros_like(unique_x, dtype=np.float64)
    counts = np.zeros_like(unique_x, dtype=np.float64)
    np.add.at(sums, inverse, y)
    np.add.at(counts, inverse, 1.0)
    return unique_x, sums / np.maximum(counts, 1.0)


def _safe_savgol_window(requested: int, n: int, polyorder: int) -> int:
    window = min(requested, n if n % 2 == 1 else n - 1)
    minimum = polyorder + 2
    if minimum % 2 == 0:
        minimum += 1
    window = max(window, minimum)
    if window > n:
        window = n if n % 2 == 1 else n - 1
    if window <= polyorder:
        raise ValueError("Not enough points for Savitzky–Golay smoothing")
    return window


def _repair_in_lambda_order(indices: np.ndarray, dense_lambda: np.ndarray) -> np.ndarray:
    order = np.argsort(dense_lambda)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    ranks = rank[indices].copy()
    K, T = len(ranks), len(order)
    if K > T:
        raise ValueError("K cannot exceed T")
    for r in range(1, K):
        ranks[r] = max(ranks[r], ranks[r - 1] + 1)
    if ranks[-1] >= T:
        ranks[-1] = T - 1
        for r in range(K - 2, -1, -1):
            ranks[r] = min(ranks[r], ranks[r + 1] - 1)
    return order[ranks]


USAGE = r'''
model_adapter = GenericTorchModelAdapter(model)

calibration, curves, reduced = build_reduced_schedule(
    model=model_adapter,
    validation_loader=val_loader,
    target_adapter=BatchTargetAdapter("z0"),
    source_adapter=BatchSourceAdapter("source_cond"),
    mask_adapter=BatchMaskAdapter("attention_mask"),
    alpha_bar_schedule=alpha_bar_schedule,  # [2000,L]
    calibration_config=CalibrationConfig(device="cuda"),
    curve_config=CurveFitConfig(method="isotonic_pchip"),
    node_config=NodeSelectionConfig(num_nodes=10),
)

save_outputs(
    calibration=calibration,
    curves=curves,
    schedule=reduced,
    output_dir="outputs/recovery_schedule_K10",
)
'''

if __name__ == "__main__":
    print(USAGE)
