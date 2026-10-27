r"""Relaxation-rate bands and the cached Jiang node set (no torch here).

Kept separate from `registry` so that plotting code can import it without
loading torch: torch's libiomp and Anaconda's MKL cannot share a process.
"""

from __future__ import annotations

import json
import os

import numpy as np

from . import jiang
from .config import DATA, L_LONG, L_TRAIN, N_STATES, TAU

T_TRAIN = L_TRAIN * TAU      # 10 s  -- horizon seen during training
T_LONG = L_LONG * TAU        # 100 s -- horizon at evaluation
S_LO_TRAIN = 1.0 / T_TRAIN   # 0.1   -- slowest rate resolvable in training
S_HI = 1.0 / TAU             # 100   -- fastest rate resolvable on the grid
ALPHA_REF = 0.55             # mid-range order used to build the shared grid

# the three models the task asks for, and the variants/ablations around them
MAIN3 = ("transformer_rope", "s4d_deep", "soe_jiang")
ABLATIONS = ("transformer_nope", "transformer_alibi", "s4d_lin_log",
             "s4d_lin_linear", "s4d_lin_narrow", "s4d_lin_softplus",
             "s4d_lin_log_longtrain", "s4d_lin_narrow_longtrain",
             "soe_jiang_table", "soe_jiang_trainband")


def jiang_nodes(T: float, n_max: int = N_STATES, alpha_ref: float = ALPHA_REF):
    """Cached Jiang node set for the band [tau, T] (shared across alpha).

    The alpha-dependence of the node set is weak (it enters only through the
    Gauss-Jacobi nodes on [0, s_min]); `src.validate` checks that one shared
    grid still fits every alpha's kernel to <= 2e-8.
    """
    path = os.path.join(DATA, f"jiang_nodes_T{T:g}_n{n_max}.npz")
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        return d["s"], json.loads(str(d["info"]))
    s, info = jiang.nodes_for_alpha(alpha_ref, TAU, T, n_max)
    info = {k: v for k, v in info.items() if k not in ("s", "w")}
    info.update(T=T, n_max=n_max, alpha_ref=alpha_ref)
    np.savez(path, s=s, info=json.dumps(info))
    return s, info
