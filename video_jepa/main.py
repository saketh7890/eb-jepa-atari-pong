"""
Video JEPA Training Script — Pong NPZ, 10 epochs.
"""

import os
from pathlib import Path

import fire
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
from omegaconf import OmegaConf

from eb_jepa.architectures import (
    DetHead,
    Projector,
    ResNet5,
    ResUNet,
    StateOnlyPredictor,
)
from eb_jepa.image_decoder import ImageDecoder
from eb_jepa.jepa import JEPA, JEPAProbe
from eb_jepa.logging import get_logger
from eb_jepa.losses import SquareLossSeq, VCLoss
from eb_jepa.training_utils import (
    get_default_dev_name,
    get_exp_name,
    get_unified_experiment_dir,
    load_checkpoint,
    load_config,
    log_config,
    log_data_info,
    log_model_info,
    save_checkpoint,
    setup_device,
    setup_seed,
    setup_wandb,
)
from examples.video_jepa.eval import (
    encode_sequence,
    safe_recon_loss,
    safe_det_loss,
    validation_loop,
)

logger = get_logger(__name__)

BYPASS_UNROLL = True


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class PongNPZSequenceDataset(Dataset):
    def __init__(self, npz_path: str, T: int = 6):
        super().__init__()
        data = np.load(npz_path, allow_pickle=False)
        obs  = data["observations"]

        assert obs.ndim == 4, f"Expected 4-D observations, got {obs.shape}"
        N, a, b, c = obs.shape

        if a <= 4 and b > 4:
            layout = "channels_first (N,C,H,W)"
        elif c <= 4 and b > 4:
            obs    = obs.transpose(0, 3, 1, 2)
            layout = "channels_last -> transposed to (N,C,H,W)"
        else:
            raise ValueError(f"Cannot detect channel axis in obs shape {obs.shape}.")

        obs = obs.astype(np.float32)
        if obs.max() > 1.5:
            obs /= 255.0

        self.obs = obs
        self.T   = T
        _, self.C, self.H, self.W = obs.shape

        logger.info(f"PongNPZSequenceDataset | {layout} | final (N,C,H,W)={obs.shape}")

        terminals = (
            data["terminals"][:N].astype(bool)
            if "terminals" in data
            else np.zeros(N, dtype=bool)
        )
        valid = [i for i in range(N - T) if not terminals[i : i + T - 1].any()]
        self.indices = np.array(valid, dtype=np.int64)
        logger.info(f"PongNPZSequenceDataset | {len(self.indices):,} clips | T={T}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        s     = self.indices[idx]
        clip  = self.obs[s : s + self.T]
        video = torch.from_numpy(clip.copy())
        loc   = torch.zeros(self.T, 1, self.H, self.W)
        return {"video": video, "digit_location": loc}


# ──────────────────────────────────────────────────────────────────────────────
# VCLoss return signature helper
# ──────────────────────────────────────────────────────────────────────────────

def call_regularizer(regularizer, tokens):
    result = regularizer(tokens, tokens)
    if not isinstance(result, (tuple, list)):
        return result, {}
    n = len(result)
    if n == 2:
        return result[0], result[1]
    elif n == 3:
        regl     = result[0]
        regldict = next((r for r in result[1:] if isinstance(r, dict)), {})
        return regl, regldict
    else:  # 4+
        regl     = result[0]
        regldict = result[3] if isinstance(result[3], dict) else {}
        return regl, regldict


# ──────────────────────────────────────────────────────────────────────────────
# Manual JEPA forward
# ──────────────────────────────────────────────────────────────────────────────

def manual_jepa_forward(jepa, x, nsteps, device):
    """
    Replaces jepa.unroll() for (B, T, C, H, W) inputs.
    Projector operates per spatial location: (B*T,D,Hs,Ws) → (B*T*Hs*Ws, D).
    """
    B, T, C, H, W = x.shape

    x_flat  = x.reshape(B * T, C, H, W)
    s_flat  = jepa.encoder(x_flat)                        # (B*T, D, Hs, Ws)
    BT, D, Hs, Ws = s_flat.shape
    states  = s_flat.reshape(B, T, D, Hs, Ws)

    context  = states[:, :2]
    n_target = min(nsteps, T - 2)
    targets  = states[:, 2 : 2 + n_target]

    ctx_in    = context.reshape(B, 2 * D, Hs, Ws)
    pred_list = []
    for _ in range(n_target):
        pred = jepa.predictor.predictor(ctx_in)
        pred_list.append(pred)
        ctx_in = torch.cat([ctx_in[:, D:], pred], dim=1)
    predictions = torch.stack(pred_list, dim=1)

    proj   = jepa.predcost.proj
    pl_sum = torch.tensor(0.0, device=device)
    for t in range(n_target):
        p_tok  = predictions[:, t].permute(0,2,3,1).reshape(-1, D)
        t_tok  = targets[:, t].permute(0,2,3,1).reshape(-1, D)
        pl_sum = pl_sum + F.mse_loss(proj(p_tok), proj(t_tok).detach())
    pl = pl_sum / max(1, n_target)

    all_tokens     = s_flat.permute(0,2,3,1).reshape(-1, D)
    regl, regldict = call_regularizer(jepa.regularizer, all_tokens)

    jepa_loss = regl + pl
    return jepa_loss, regl, regldict, pl


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def run(
    fname: str = "examples/video_jepa/cfgs/default.yaml",
    cfg=None,
    folder: str = None,
    epochs: int = 10,
    **overrides,
):
    """
    Train Video JEPA on Pong NPZ for 10 epochs.

    Usage:
        python -m examples.video_jepa.main --meta.device=cuda
        python -m examples.video_jepa.main --meta.device=cuda --data.batch_size=32
    """
    if cfg is None:
        cfg = load_config(fname, overrides if overrides else None)

    cfg.optim.epochs = 10
    cfg.model.dobs   = 3

    weight_decay = cfg.optim.get("weight_decay", 1e-4)
    clip_norm    = cfg.optim.get("clip_norm",    1.0)
    log_every    = cfg.logging.get("log_every",  1)
    save_every   = cfg.logging.get("save_every", 10)

    device = setup_device(cfg.meta.device)
    setup_seed(cfg.meta.seed)

    if folder is None:
        if cfg.meta.get("model_folder"):
            exp_dir  = Path(cfg.meta.model_folder)
            exp_name = exp_dir.name.rsplit("_seed", 1)[0]
        else:
            exp_name = get_exp_name("video_jepa", cfg)
            exp_dir  = get_unified_experiment_dir(
                example_name="video_jepa",
                sweep_name=get_default_dev_name(),
                exp_name=exp_name,
                seed=cfg.meta.seed,
            )
    else:
        exp_dir  = Path(folder)
        exp_dir.mkdir(parents=True, exist_ok=True)
        exp_name = exp_dir.name.rsplit("_seed", 1)[0]

    wandb_run = setup_wandb(
        project="eb_jepa",
        config={"example": "video_jepa", **OmegaConf.to_container(cfg, resolve=True)},
        run_dir=exp_dir, run_name=exp_name,
        tags=["video_jepa", f"seed_{cfg.meta.seed}", "pong"],
        group=cfg.logging.get("wandb_group"),
        enabled=cfg.logging.log_wandb,
        sweep_id=cfg.logging.get("wandb_sweep_id"),
    )

    npz_path = f"/scratch/{os.environ['USER']}/pong_10k.npz"
    logger.info(f"Loading Pong dataset from: {npz_path}")

    train_set    = PongNPZSequenceDataset(npz_path, T=6)
    val_set      = PongNPZSequenceDataset(npz_path, T=6)
    train_loader = DataLoader(train_set, batch_size=cfg.data.batch_size,
                              shuffle=True,  num_workers=cfg.data.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=cfg.data.batch_size,
                              shuffle=False, num_workers=cfg.data.num_workers, pin_memory=True)
    log_data_info("Pong-NPZ", len(train_loader), cfg.data.batch_size,
                  train_samples=len(train_set), val_samples=len(val_set))

    logger.info(f"Building model with dobs={cfg.model.dobs}")
    assert cfg.model.dobs == 3

    encoder         = ResNet5(cfg.model.dobs, cfg.model.henc, cfg.model.dstc)
    predictor_model = ResUNet(2 * cfg.model.dstc, cfg.model.hpre, cfg.model.dstc)
    predictor       = StateOnlyPredictor(predictor_model, context_length=2)
    projector       = Projector(f"{cfg.model.dstc}-{cfg.model.dstc*4}-{cfg.model.dstc*4}")
    regularizer     = VCLoss(cfg.loss.std_coeff, cfg.loss.cov_coeff, proj=projector)
    ploss           = SquareLossSeq(projector)
    jepa            = JEPA(encoder, encoder, predictor, regularizer, ploss).to(device)

    decoder         = ImageDecoder(cfg.model.dstc, cfg.model.dobs)
    dethead         = DetHead(cfg.model.dstc, cfg.model.hpre, cfg.model.dobs)
    pixel_decoder   = JEPAProbe(jepa, decoder, nn.MSELoss()).to(device)
    detection_head  = JEPAProbe(jepa, dethead, nn.BCELoss()).to(device)

    assert encoder.conv1.in_channels == 3
    logger.info(f"Encoder first Conv2d: in_channels={encoder.conv1.in_channels} ✓")

    log_model_info(jepa, {
        "encoder":   sum(p.numel() for p in encoder.parameters()),
        "predictor": sum(p.numel() for p in predictor.parameters()),
    })

    jepa.train(); detection_head.train(); pixel_decoder.train()

    optimizer = Adam([
        {"params": jepa.parameters(),
         "lr": cfg.optim.lr,      "weight_decay": weight_decay},
        {"params": pixel_decoder.head.parameters(),
         "lr": cfg.optim.lr / 10, "weight_decay": weight_decay},
        {"params": detection_head.head.parameters(),
         "lr": cfg.optim.lr,      "weight_decay": weight_decay},
    ])
    scaler    = GradScaler("cuda")
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.optim.epochs, eta_min=1e-6)

    log_config(cfg)

    start_epoch = 0; global_step = 0; best_val = float("inf")
    if cfg.meta.get("load_model"):
        ckpt_info   = load_checkpoint(exp_dir / "latest.pth.tar", jepa, optimizer, device=device)
        start_epoch = ckpt_info.get("epoch", 0)
        global_step = ckpt_info.get("step",  0)

    logger.info(
        f"Starting {cfg.optim.epochs} epochs | "
        f"lr={cfg.optim.lr} | wd={weight_decay} | BYPASS_UNROLL={BYPASS_UNROLL}"
    )

    for epoch in range(start_epoch, cfg.optim.epochs):
        jepa.train(); detection_head.train(); pixel_decoder.train()

        pbar       = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.optim.epochs}")
        epoch_loss = 0.0

        for batch in pbar:
            batch   = {k: v.to(device) for k, v in batch.items()}
            x       = batch["video"]           # (B, T, C, H, W)
            loc_map = batch["digit_location"]  # (B, T, 1, H, W)

            if global_step == 0:
                logger.info(f"First batch (B,T,C,H,W) = {tuple(x.shape)}")
                assert x.shape[2] == 3, f"Expected C=3, got shape {tuple(x.shape)}"

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda"):
                jepa_loss, regl, regldict, pl = manual_jepa_forward(
                    jepa, x, cfg.model.steps, device
                )
                states_grad = encode_sequence(jepa, x)        # (B,T,D,Hs,Ws)
                recon_loss  = safe_recon_loss(pixel_decoder, states_grad, x)
                det_loss    = safe_det_loss(detection_head, states_grad, loc_map)
                total_loss  = jepa_loss + recon_loss + det_loss

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(jepa.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss  += total_loss.item()
            global_step += 1

            pbar.set_postfix({
                "total": f"{total_loss.item():.4f}",
                "jepa":  f"{jepa_loss.item():.4f}",
                "vc":    f"{regl.item():.4f}",
                "pred":  f"{pl.item():.4f}",
                "recon": f"{recon_loss.item():.4f}",
            })

        avg_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} avg loss: {avg_loss:.6f}")

        if epoch % log_every == 0:
            val_logs = validation_loop(
                val_loader, jepa, detection_head,
                pixel_decoder, cfg.model.steps, device,
            )
            val_metric = val_logs.get("val/recon_loss", avg_loss)

            if wandb_run:
                import wandb
                wandb.log({"epoch": epoch, "train/loss": avg_loss, **val_logs},
                          step=global_step)

            if val_metric < best_val:
                logger.info(f"Val improved: {best_val:.6f} → {val_metric:.6f} ✓")
                best_val = val_metric
                save_checkpoint(exp_dir / "best.pth.tar", model=jepa,
                                optimizer=optimizer, epoch=epoch, step=global_step)

            jepa.train(); detection_head.train(); pixel_decoder.train()

        save_checkpoint(exp_dir / "latest.pth.tar", model=jepa,
                        optimizer=optimizer, epoch=epoch, step=global_step)
        if epoch % save_every == 0 and epoch > 0:
            save_checkpoint(exp_dir / f"epoch_{epoch}.pth.tar", model=jepa,
                            optimizer=optimizer, epoch=epoch, step=global_step)

        scheduler.step()

    if wandb_run:
        import wandb; wandb.finish()

    logger.info("Training complete!")
    return exp_dir


if __name__ == "__main__":
    fire.Fire(run)