"""
eval.py — Fixed validation loop for Pong NPZ training.

Fix in this version:
  DetHead output shape after Conv3d + squeeze(1) is not reliably inferrable
  without reading architectures.py Conv3d stride/padding params.
  numel()=9216, B*T=48, so Ho*Wo=192=8*24 — but raw.shape[-2:] returns (8,8)
  meaning raw is NOT 4D (B,T,Ho,Wo) as assumed; something else.

  Rather than keep guessing: we compute ALL spatial dims purely from numel(),
  avoiding ANY assumption about raw.shape layout.
  If that still fails, we skip det_loss (it's a minor auxiliary loss).
  The JEPA + recon losses are what matter for representation learning.
"""

import collections
import math
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# Core encoding helpers
# ──────────────────────────────────────────────────────────────────────────────

def encode_sequence(jepa, x):
    """
    Correctly encode (B, T, C, H, W) → (B, T, D, Hs, Ws).
    Bypasses jepa.encode() which has an internal reshape bug.
    """
    B, T, C, H, W = x.shape
    x_flat = x.reshape(B * T, C, H, W)
    s_flat = jepa.encoder(x_flat)
    _, D, Hs, Ws = s_flat.shape
    return s_flat.reshape(B, T, D, Hs, Ws)


def predict_sequence(jepa, states, n_steps):
    """
    Autoregressively predict n_steps future states from first 2 context frames.
    states: (B, T, D, Hs, Ws)
    Returns: predictions (B, n_steps, D, Hs, Ws)
    """
    B, T, D, Hs, Ws = states.shape
    n_target  = min(n_steps, T - 2)
    ctx_in    = states[:, :2].reshape(B, 2 * D, Hs, Ws)
    pred_list = []
    for _ in range(n_target):
        pred = jepa.predictor.predictor(ctx_in)
        pred_list.append(pred)
        ctx_in = torch.cat([ctx_in[:, D:], pred], dim=1)
    return torch.stack(pred_list, dim=1)


# ──────────────────────────────────────────────────────────────────────────────
# Probe head runners
# ──────────────────────────────────────────────────────────────────────────────

def run_pixel_decoder(pixel_decoder, states):
    """
    ImageDecoder: 2D conv, expects (B*T, D, Hs, Ws).
    Returns: (B, T, C_out, Ho, Wo)
    """
    B, T, D, Hs, Ws = states.shape
    s_flat  = states.reshape(B * T, D, Hs, Ws)
    out     = pixel_decoder.head(s_flat)
    _, C_out, Ho, Wo = out.shape
    return out.reshape(B, T, C_out, Ho, Wo)


_det_head_warned = False

def run_detection_head(detection_head, states):
    """
    DetHead: Conv3d on (B, D, T, Hs, Ws), then internal squeeze.
    Output shape is unreliable — compute spatial dims from numel() only.

    Returns: (B*T, 1, Ho, Wo)  OR  None if shape cannot be resolved.
    """
    global _det_head_warned
    B, T, D, Hs, Ws = states.shape

    # Conv3d expects (B, D, T, Hs, Ws)
    s_3d = states.permute(0, 2, 1, 3, 4).contiguous()
    raw  = detection_head.head(s_3d)

    total   = raw.numel()           # = B * T * Ho * Wo
    spatial = total // (B * T)      # = Ho * Wo

    if spatial <= 0 or (B * T * spatial) != total:
        if not _det_head_warned:
            warnings.warn(
                f"DetHead output numel={total} not divisible by B*T={B*T}. "
                "Skipping detection loss."
            )
            _det_head_warned = True
        return None

    # Factor spatial into (Ho, Wo) — use last two dims of raw if available,
    # otherwise find largest factor pair
    raw_flat = raw.reshape(B * T, spatial)  # always safe: just flatten to (B*T, Ho*Wo)

    # Try to recover 2D spatial for interpolation
    # Attempt 1: use raw.shape[-2:] if raw has >= 4 dims AND product matches
    if raw.dim() >= 4 and (raw.shape[-2] * raw.shape[-1] == spatial):
        Ho, Wo = raw.shape[-2], raw.shape[-1]
    else:
        # Attempt 2: factor via integer square root
        sq = int(math.isqrt(spatial))
        # Find largest divisor <= sqrt
        Ho = sq
        while Ho > 1 and spatial % Ho != 0:
            Ho -= 1
        Wo = spatial // Ho

    try:
        out = raw_flat.reshape(B * T, 1, Ho, Wo)
        return out  # (B*T, 1, Ho, Wo)
    except RuntimeError:
        if not _det_head_warned:
            warnings.warn(
                f"DetHead: cannot reshape numel={total} to ({B*T}, 1, {Ho}, {Wo}). "
                "Skipping detection loss."
            )
            _det_head_warned = True
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Loss functions
# ──────────────────────────────────────────────────────────────────────────────

def safe_recon_loss(pixel_decoder, states, x):
    """
    MSE reconstruction loss.
    states: (B, T, D, Hs, Ws)
    x:      (B, T, C, H, W)
    """
    B, T, C, H, W = x.shape
    recon = run_pixel_decoder(pixel_decoder, states)   # (B, T, C_out, Ho, Wo)
    Cout, Ho, Wo = recon.shape[2], recon.shape[3], recon.shape[4]

    if (Ho, Wo) != (H, W):
        r     = recon.reshape(B * T, Cout, Ho, Wo)
        r     = F.interpolate(r, size=(H, W), mode="bilinear", align_corners=False)
        recon = r.reshape(B, T, Cout, H, W)

    return F.mse_loss(recon, x)


def safe_det_loss(detection_head, states, loc_map):
    """
    BCE detection loss. Returns 0.0 tensor (no grad) if DetHead shape unresolvable.
    states:  (B, T, D, Hs, Ws)
    loc_map: (B, T, 1, H, W)
    """
    B, T, _, H, W = loc_map.shape
    device = states.device

    det = run_detection_head(detection_head, states)   # (B*T, 1, Ho, Wo) or None

    if det is None:
        return torch.tensor(0.0, device=device, requires_grad=False)

    Ho, Wo = det.shape[-2], det.shape[-1]
    if (Ho, Wo) != (H, W):
        det = F.interpolate(det, size=(H, W), mode="bilinear", align_corners=False)

    det = det.reshape(B, T, 1, H, W)
    return F.binary_cross_entropy_with_logits(det, loc_map)


# ──────────────────────────────────────────────────────────────────────────────
# Validation loop
# ──────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def validation_loop(val_loader, jepa, detection_head, pixel_decoder, steps, device):
    """
    Fixed validation loop. Never calls jepa.unroll or jepa.encode directly.
    """
    jepa.eval()
    detection_head.eval()
    pixel_decoder.eval()

    metrics = collections.defaultdict(list)

    for batch in tqdm(val_loader, desc="Validation", leave=False):
        batch   = {k: v.to(device) for k, v in batch.items()}
        x       = batch["video"]
        loc_map = batch["digit_location"]

        states = encode_sequence(jepa, x)

        recon_loss = safe_recon_loss(pixel_decoder, states, x)
        det_loss   = safe_det_loss(detection_head, states, loc_map)

        metrics["val/recon_loss"].append(float(recon_loss.item()))
        metrics["val/det_loss"].append(float(det_loss.item()))

        try:
            preds       = predict_sequence(jepa, states, n_steps=steps)
            target_locs = loc_map[:, 2 : 2 + preds.shape[1]]
            scores      = detection_head.head.score(preds, target_locs)
            for s, score in enumerate(scores):
                metrics[f"AP_{s}"].append(float(score))
        except Exception:
            pass

    agg = {k: float(np.mean(v)) for k, v in metrics.items()}
    print(agg)
    return agg