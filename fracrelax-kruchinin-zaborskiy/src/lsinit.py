r"""Least-squares initialisation of the readout of the single-layer SSMs.

Why.  `soe_jiang` has the right poles but still ends at ~1e-3 relative error,
while Jiang nodes + *least-squares* weights reach ~3e-8 (results/tables.md, last
row of the accuracy table).  So the 1e-3 floor is a property of how the readout
is *optimised* (Adam on a badly conditioned problem: the optimal c_j are
O(1..30) with a huge dynamic range), not of the architecture.  This module starts
the same model AT the least-squares solution so that training can only keep it,
or lose it, and the question "is the gap an optimisation artefact?" gets an
answer.

What is set.  For each training alpha_i the exact L1 impulse response is fitted
(`jiang.fit_weights_to_kernel`) with the model's CURRENT rates, giving c(alpha_i).
  * readout="table": the table rows are set to c(alpha_i).
  * readout="mlp":   the final linear layer is set by least squares so that
                     to_c(emb(alpha_i)) = c(alpha_i) at the 8 training orders
                     (exact interpolation; behaviour between them is whatever
                     the embedding gives, and is what training then shapes).
The fit uses only `L_fit` lags, i.e. only the horizon the caller is allowed to
know: L_LONG for `soe_jiang_lsinit` (whose nodes already know T=100) and L_TRAIN
for `s4d_lin_log_lsinit` (which does not).
"""

from __future__ import annotations

import numpy as np
import torch

from . import jiang
from .config import ALPHAS_TRAIN, LAM, TAU
from .l1 import discrete_kernel


def ls_readout(rates: np.ndarray, alphas, L_fit: int, tau: float = TAU,
               lam: float = LAM) -> np.ndarray:
    """(n_alpha, n_exp) least-squares coefficients for the given rates."""
    out = []
    for a in alphas:
        g = discrete_kernel(float(a), lam, tau, L_fit)
        out.append(jiang.fit_weights_to_kernel(np.asarray(rates, np.float64), g, tau))
    return np.stack(out)


@torch.no_grad()
def apply_ls_init(model, L_fit: int, alphas=ALPHAS_TRAIN) -> dict:
    """Overwrite the readout of `model` (a _SOEBase) with the LS solution."""
    C = ls_readout(model.rates().detach().cpu().numpy(), alphas, L_fit)
    Ct = torch.as_tensor(C, dtype=torch.float64)
    if model.readout == "table":
        model.c.copy_(Ct.to(model.c.dtype))
    else:
        dev = model.to_c.weight.device
        a = torch.as_tensor(alphas, dtype=torch.float32, device=dev)
        E = torch.stack([model.emb(a[i:i + 1])[0] for i in range(len(a))])
        X = torch.cat([E.double(), torch.ones(len(a), 1, dtype=torch.float64,
                                              device=dev)], dim=1)       # (n_a, d+1)
        sol = torch.linalg.lstsq(X.cpu(), Ct, rcond=1e-10,
                                 driver="gelsd").solution               # (d+1, n_exp)
        model.to_c.weight.copy_(sol[:-1].T.to(model.to_c.weight))
        model.to_c.bias.copy_(sol[-1].to(model.to_c.bias))
    return dict(L_fit=L_fit, c_abs_max=float(np.abs(C).max()))
