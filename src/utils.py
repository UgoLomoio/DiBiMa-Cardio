"""Slim copy of DiBiMa-Cardio src/utils.py: only QRSAwareMSELoss (imported by src/models.py)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QRSAwareMSELoss(nn.Module):
    """
    QRS-aware MSE loss for ECG imputation.

    Uses a Gaussian weight map centered on R-peaks to upweight
    reconstruction error around the QRS complex, while keeping a
    baseline MSE over the whole signal.

    Expected inputs:
      - pred:   Tensor [B, T] or [B, 1, T] (imputed ECG)
      - target: Tensor [B, T] or [B, 1, T] (ground-truth ECG)
      - r_peak_idx: list[list[int]] or list[np.ndarray]
          r_peak_idx[b] = 1D array-like of R-peak sample indices for batch item b

    Hyperparameters:
      - sigma: width (in samples) of the Gaussian around each R-peak
      - beta:  additional weight at the peak center (beta=0 => plain MSE)
      - eps:   small constant for numerical stability (unused in current formula)
    """

    def __init__(self, sigma: int = 8, beta: float = 5.0, eps: float = 1e-8):
        super().__init__()
        self.sigma = sigma
        self.beta = beta
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, r_peak_idx) -> torch.Tensor:
        # Normalize shapes to [B, T]
        if pred.dim() == 3 and pred.shape[1] == 1:
            pred = pred.squeeze(1)
        if target.dim() == 3 and target.shape[1] == 1:
            target = target.squeeze(1)

        if pred.shape != target.shape or pred.dim() != 2:
            raise ValueError("pred and target must be [B, T] or [B, 1, T] tensors.")

        B, T = pred.shape
        device = pred.device

        # ---- Enforce non-empty R-peaks ----
        if not isinstance(r_peak_idx, (list, tuple)):
            raise TypeError("r_peak_idx must be a list or tuple of length B.")
        if len(r_peak_idx) != B:
            raise ValueError(f"r_peak_idx length ({len(r_peak_idx)}) must match batch size ({B}).")

        # Base weights = 1 everywhere
        w = torch.ones_like(pred)

        # Add Gaussian bumps around each R-peak
        t_grid = torch.arange(T, device=device).view(1, -1)  # [1, T]

        for b in range(B):
            if len(r_peak_idx[b]) != 0:
                peaks = r_peak_idx[b]
                # At this point peaks is guaranteed non-empty by the checks above,
                # but we still filter out-of-range indices.
                peaks = [int(p) for p in peaks if 0 <= int(p) < T]
                if len(peaks) == 0:
                    raise ValueError(
                        f"QRSAwareMSELoss: all R-peaks for batch index {b} are out of bounds [0, {T})."
                    )

                peaks_tensor = torch.tensor(peaks, device=device).view(-1, 1)  # [N, 1]
                # Gaussian for each peak, then sum over peaks
                gauss = torch.exp(-0.5 * ((t_grid - peaks_tensor) / self.sigma) ** 2)  # [N, T]
                w[b] += self.beta * gauss.sum(dim=0)  # [T]
            else:
                #just use MSELoss if no R-peaks are provided for this batch item
                w[b] = torch.ones(T, device=device)

        # Weighted MSE
        err = (pred - target) ** 2
        loss = (w * err).mean()

        return loss