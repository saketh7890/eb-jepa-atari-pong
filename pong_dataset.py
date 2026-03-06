
import numpy as np
import torch
from torch.utils.data import Dataset

class PongNPZSequenceDataset(Dataset):
    """
    Builds fixed-length frame sequences from pong_10k.npz.
    Expects keys: observations, next_observations, actions, rewards (dones optional).
    Returns a tensor of shape [T, C, H, W] with values in [0, 1].
    """
    def __init__(self, npz_path: str, T: int = 6):
        self.path = npz_path
        self.T = T
        self.d = np.load(npz_path)

        obs = self.d["observations"]
        self.obs = obs
        self.N = len(obs)

        # Use dones if present to avoid sequences crossing episode boundaries
        if "dones" in self.d.files:
            self.dones = self.d["dones"].astype(bool)
        elif "terminals" in self.d.files:
            self.dones = self.d["terminals"].astype(bool)
        else:
            self.dones = None

        # valid start indices
        self.starts = []
        for i in range(self.N - self.T):
            if self.dones is None:
                self.starts.append(i)
            else:
                # if any done occurs inside the window, skip
                if not self.dones[i:i+self.T].any():
                    self.starts.append(i)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        i = self.starts[idx]
        x = self.obs[i:i+self.T]  # [T, H, W, C] or [T, ...]
        x = torch.from_numpy(x)

        # common Atari format: uint8 images [H,W,C]
        if x.dtype != torch.float32:
            x = x.float()

        # If images are uint8 0..255 -> normalize
        if x.max() > 1.5:
            x = x / 255.0

        # Convert to [T, C, H, W] if it looks like [T, H, W, C]
        if x.ndim == 4 and x.shape[-1] in (1, 3, 4):
            x = x.permute(0, 3, 1, 2).contiguous()

        return x


