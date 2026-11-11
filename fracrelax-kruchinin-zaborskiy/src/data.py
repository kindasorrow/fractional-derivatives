r"""Dataset loading + shared metrics."""

from __future__ import annotations

import json
import os

import numpy as np
import torch

from .config import DATA


class SeqData:
    """A whole .npz held in memory (optionally on the GPU)."""

    def __init__(self, path: str, device="cpu", max_n: int | None = None):
        d = np.load(path, allow_pickle=True)
        n = len(d["alpha"]) if max_n is None else min(max_n, len(d["alpha"]))
        sl = slice(0, n)
        self.f = torch.as_tensor(d["f"][sl], dtype=torch.float32, device=device)
        self.y = torch.as_tensor(d["y"][sl], dtype=torch.float32, device=device)
        self.alpha = torch.as_tensor(d["alpha"][sl], dtype=torch.float32, device=device)
        self.alpha_id = torch.as_tensor(d["alpha_id"][sl], dtype=torch.long, device=device)
        self.kind_id = np.asarray(d["kind_id"][sl])
        self.alphas = np.asarray(d["alphas"])
        self.length = (np.asarray(d["length"][sl]) if "length" in d.files
                       else np.full(n, self.f.shape[1], dtype=np.int32))
        self.tau = float(d["tau"])
        self.lam = float(d["lam"])
        self.meta = json.loads(str(d["meta"])) if "meta" in d.files else {}
        self.path = path

    def __len__(self):
        return self.f.shape[0]

    @property
    def L(self):
        return self.f.shape[1]


def path(name: str) -> str:
    return os.path.join(DATA, name)


def rel_l2(yhat: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Per-sequence relative L2 error ||yhat-y|| / ||y||."""
    num = torch.linalg.vector_norm(yhat - y, dim=-1)
    den = torch.linalg.vector_norm(y, dim=-1) + eps
    return num / den


def rel_mse_loss(yhat: torch.Tensor, y: torch.Tensor, eps: float = 1e-8):
    """Scale-free training loss: mean over the batch of ||e||^2/||y||^2.

    Needed because y's magnitude varies by ~1 order of magnitude across the
    forcing families, and an unweighted MSE would only fit the loudest ones.
    """
    num = ((yhat - y) ** 2).mean(dim=-1)
    den = (y ** 2).mean(dim=-1) + eps
    return (num / den).mean()
