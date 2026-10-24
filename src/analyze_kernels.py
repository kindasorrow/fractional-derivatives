r"""Second half of the task-5 analysis: the kernel each model realises.

Run after `python -m src.analyze`, in a separate process: under Anaconda,
numpy's MKL BLAS and torch's libiomp cannot both initialise in one process
(a pip venv with OpenBLAS numpy is unaffected).

Adds to results/analysis.json:
  kernels   the implied impulse response of every single-layer model, the true
            L1 kernel, and the training-free Jiang+LS reference
  spectral  the exact spectral density mu(s), for the theory overlay

Run:  python -m src.analyze_kernels
"""

from __future__ import annotations

import json
import os

import numpy as np

from . import exact, jiang
from .bands import jiang_nodes
from .config import ALPHAS_TRAIN, LAM, L_LONG, N_STATES, RESULTS, TAU
from .l1 import discrete_kernel


def implied_kernel(rates, coeffs, alpha_idx, L=L_LONG, tau=TAU):
    """The convolution kernel a learned single-layer SSM actually realises."""
    s = np.asarray(rates, dtype=np.float64)
    c = np.asarray(coeffs, dtype=np.float64)[alpha_idx]
    return jiang.soe_design(s, tau, np.arange(1, L + 1)) @ c


def _kernel_errors(name, k, k_true, n_split=1000):
    """Several views of the kernel error.

    The median relative error over all 10^4 lags is dominated by the far tail,
    where the true kernel is ~1e-6 of its peak and contributes nothing to y.
    The L2 measure is the one that actually drives the output error; the
    short/long split says whether the damage is inside or outside the horizon
    the model was trained on.
    """
    rel = np.abs(k - k_true) / k_true
    return {
        name + "_l2": float(np.linalg.norm(k - k_true) / np.linalg.norm(k_true)),
        name + "_rel_med_short": float(np.median(rel[:n_split])),
        name + "_rel_med_long": float(np.median(rel[n_split:])),
    }


def main():
    path = os.path.join(RESULTS, "analysis.json")
    with open(path) as fh:
        rep = json.load(fh)
    s_jiang_long = np.asarray(rep["jiang"]["long"]["s"])

    # ---- kernel reconstruction for the single-layer models ----------------
    print("[analyze] implied kernels vs the true L1 kernel")
    rep["kernels"] = {}
    n_plot = np.unique(np.round(np.exp(np.linspace(0, np.log(L_LONG), 300)))
                       .astype(int))
    for ai, a in enumerate(ALPHAS_TRAIN):
        g = discrete_kernel(a, LAM, TAU, L_LONG)
        entry = dict(alpha=float(a), n=n_plot.tolist(),
                     true=(TAU * g[n_plot - 1]).tolist())
        for nm in ("soe_jiang", "s4d_lin_log", "s4d_lin_linear",
                   "s4d_lin_narrow", "s4d_lin_log_longtrain"):
            d = rep["models"].get(nm)
            if d is None:
                continue
            if d.get("kind") != "single_layer":
                continue
            k = implied_kernel(d["rates"], d["coeffs"], ai)
            entry[nm] = k[n_plot - 1].tolist()
            entry.update(_kernel_errors(nm, k, TAU * g))
        # training-free ceiling
        c_ls = jiang.fit_weights_to_kernel(s_jiang_long, g, TAU)
        k_ls = jiang.soe_design(s_jiang_long, TAU, np.arange(1, L_LONG + 1)) @ c_ls
        entry["soe_ls"] = k_ls[n_plot - 1].tolist()
        entry.update(_kernel_errors("soe_ls", k_ls, TAU * g))
        rep["kernels"][f"{a:.1f}"] = entry
        print(f"    alpha={a}: " + "   ".join(
            f"{k.replace('_l2', '')} L2 {v:.2e}"
            for k, v in entry.items() if k.endswith("_l2")))

    # ---- exact spectral density (theory overlay) -------------------------
    s_grid = np.exp(np.linspace(np.log(1e-5), np.log(3e3), 400))
    rep["spectral"] = {f"{a:.1f}": dict(
        s=s_grid.tolist(),
        mu=exact.spectral_density(s_grid, a, LAM).tolist(),
        s_mu=(s_grid * exact.spectral_density(s_grid, a, LAM)).tolist())
        for a in (0.2, 0.5, 0.9)}


    with open(path, "w") as fh:
        json.dump(rep, fh, indent=2)
    print("wrote results/analysis.json (kernels + spectral added)")


if __name__ == "__main__":
    main()
