r"""CPU reference solvers, benchmarked without torch in the process.

Separate from `src.bench` because under Anaconda torch's libiomp and the MKL
behind numpy cannot be initialised in the same process (OMP error #15); a pip
venv with OpenBLAS numpy is unaffected, but the split keeps both usable.

  l1_direct  the O(L^2) L1 recursion that produced the data
  l1_fft     the same scheme as a convolution, given its impulse response
  soe_ls     Jiang nodes + least-squares weights run as a 64-mode recursion:
             the classical fast solver, and the training-free ceiling of the
             fixed-exponent architecture

Run:  python -m src.bench_classical
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from . import jiang
from .bands import T_LONG, jiang_nodes
from .config import LAM, L_SCALING, N_STATES, RESULTS, TAU
from .l1 import discrete_kernel, l1_solve, l1_solve_fft


def path(name):
    from .config import DATA
    return os.path.join(DATA, name)


# ---------------------------------------------------------------------------
# classical references (CPU / numpy)
# ---------------------------------------------------------------------------
def classical_bench(alpha=0.5, lam=LAM, tau=TAU, lengths=L_SCALING, reps=3):
    rows = []
    s_nodes, _ = jiang_nodes(T_LONG, N_STATES)
    for L in lengths:
        f = np.random.default_rng(0).standard_normal(L)
        g = discrete_kernel(alpha, lam, tau, L)
        c = jiang.fit_weights_to_kernel(s_nodes, g, tau)
        y_ref = l1_solve(f, alpha, lam, tau)

        t0 = time.perf_counter()
        for _ in range(reps):
            l1_solve(f, alpha, lam, tau)
        t_direct = (time.perf_counter() - t0) / reps

        t0 = time.perf_counter()
        for _ in range(reps):
            l1_solve_fft(f, alpha, lam, tau, kernel=g)
        t_fft = (time.perf_counter() - t0) / reps

        t0 = time.perf_counter()
        for _ in range(reps):
            y_soe = jiang.soe_apply(s_nodes, c, f, tau)
        t_soe = (time.perf_counter() - t0) / reps

        rows.append(dict(L=L,
                         l1_direct_ms=t_direct * 1e3,
                         l1_fft_ms=t_fft * 1e3,
                         soe_ls_ms=t_soe * 1e3,
                         soe_ls_rel_l2=float(np.linalg.norm(y_soe - y_ref) /
                                             np.linalg.norm(y_ref)),
                         l1_direct_mem_MiB=L * 8 / 2 ** 20 * 2,
                         soe_mem_MiB=(len(s_nodes) + L) * 8 / 2 ** 20))
        print(f"  classical L={L:6d}  l1_direct {t_direct * 1e3:8.1f} ms  "
              f"l1_fft {t_fft * 1e3:6.2f} ms  soe {t_soe * 1e3:6.2f} ms  "
              f"soe relL2 {rows[-1]['soe_ls_rel_l2']:.2e}")
    return rows


def soe_ls_accuracy(lengths=L_SCALING, lam=LAM, tau=TAU):
    """Training-free SOE (Jiang nodes + LS weights) on the real test sets."""
    s_nodes, _ = jiang_nodes(T_LONG, N_STATES)
    out = {}
    for L in lengths:
        ds = np.load(path(f"scale_L{L}.npz"))
        f, y, aid = ds["f"], ds["y"], ds["alpha_id"]
        errs = np.empty(len(f))
        for ai, a in enumerate(ds["alphas"]):
            m = aid == ai
            if not m.any():
                continue
            g = discrete_kernel(float(a), lam, tau, L)
            c = jiang.fit_weights_to_kernel(s_nodes, g, tau)
            yh = jiang.soe_apply(s_nodes, c, f[m].astype(np.float64), tau)
            errs[m] = (np.linalg.norm(yh - y[m], axis=1) /
                       np.linalg.norm(y[m], axis=1))
        out[L] = dict(med=float(np.median(errs)), mean=float(errs.mean()),
                      p90=float(np.percentile(errs, 90)))
        print(f"  soe_ls  L={L:6d}  med relL2 {out[L]['med']:.3e}")
    return out




def main():
    out = dict(classical=classical_bench(),
               soe_ls={str(k): v for k, v in soe_ls_accuracy().items()})
    # a wider sweep, to show the asymptotic exponents: inside 1e3..1e4 the L1
    # recursion is still dominated by per-step python/BLAS call overhead and
    # only looks like O(L^1.2).
    print("extended sweep (scaling exponents only)")
    out["classical_extended"] = classical_bench(
        lengths=(1000, 2000, 4000, 8000, 16000, 32000, 64000), reps=2)
    with open(os.path.join(RESULTS, "bench_classical.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote results/bench_classical.json")


if __name__ == "__main__":
    main()
