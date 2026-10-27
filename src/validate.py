r"""Numerical validation of the solver, the exact references and Jiang's rule.

Run:  python -m src.validate
Writes results/validation.json and results/fig_validation.png
"""

from __future__ import annotations

import json
import os

import numpy as np

from . import exact, jiang
from .config import LAM, RESULTS, TAU
from .l1 import discrete_kernel, l1_solve, l1_solve_fft


def check_spectral():
    out = {}
    for a in (0.2, 0.5, 0.9):
        out[f"alpha={a}"] = exact.check_spectral_density(a, LAM)
    return out


def check_l1_convergence(alpha=0.5, lam=1.0, T=1.0):
    """f == 1: compare the L1 solution at t=T with the spectral exact value."""
    ref = float(exact.step_response(T, alpha, lam)[0])
    rows = []
    for N in (100, 200, 400, 800, 1600, 3200, 6400):
        tau = T / N
        y = l1_solve(np.ones(N), alpha, lam, tau)
        rows.append((N, tau, float(y[-1]), abs(y[-1] - ref) / abs(ref)))
    orders = [np.log2(rows[i][3] / rows[i + 1][3]) for i in range(len(rows) - 1)]
    return dict(reference=ref,
                table=[dict(N=r[0], tau=r[1], y=r[2], rel_err=r[3]) for r in rows],
                observed_orders=[float(o) for o in orders])


def check_lti(alpha=0.35, lam=LAM, tau=TAU, L=3000, seed=0):
    rng = np.random.default_rng(seed)
    f = rng.standard_normal((4, L))
    y1 = l1_solve(f, alpha, lam, tau)
    y2 = l1_solve_fft(f, alpha, lam, tau)
    return float(np.max(np.abs(y1 - y2)) / np.max(np.abs(y1)))


def check_discrete_kernel(alpha=0.5, lam=LAM, tau=TAU, L=10000):
    g = discrete_kernel(alpha, lam, tau, L)
    n = np.unique(np.round(np.exp(np.linspace(0, np.log(L), 40))).astype(int))
    t = n * tau
    k = exact.kernel(t, alpha, lam)
    rel = np.abs(g[n - 1] - k) / k
    return dict(t=t.tolist(), g=g[n - 1].tolist(), K=k.tolist(),
                rel_err=rel.tolist(), max_rel_err_for_t_gt_10tau=float(
                    np.max(rel[t > 10 * tau])))


def check_jiang(tau=TAU, T=100.0):
    rows = []
    for alpha in (0.2, 0.5, 0.9):
        beta = 1.0 - alpha
        for eps in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10):
            s, w = jiang.soe_power(beta, tau, T, eps=eps, n_jacobi=16, n_leg=8)
            err, _, _ = jiang.soe_power_error(beta, s, w, tau, T)
            rows.append(dict(alpha=alpha, beta=beta, eps=eps, n_exp=int(s.size),
                             max_rel_err=err,
                             s_min=float(s.min()), s_max=float(s.max())))
    return rows


def check_soe_fit(n_max=64, tau=TAU, T=100.0, lam=LAM):
    """Jiang nodes + least-squares weights vs the true discrete L1 kernel."""
    L = int(round(T / tau))
    rows = []
    for alpha in (0.2, 0.3, 0.5, 0.7, 0.9):
        s, info = jiang.nodes_for_alpha(alpha, tau, T, n_max)
        g = discrete_kernel(alpha, lam, tau, L)
        c = jiang.fit_weights_to_kernel(s, g, tau)
        n = np.arange(1, L + 1)
        appr = jiang.soe_design(s, tau, n) @ c
        rel = np.abs(appr - tau * g) / (tau * g)
        rows.append(dict(alpha=alpha, n_exp=int(s.size),
                         power_rule_err=info["err"],
                         kernel_max_rel_err=float(rel.max()),
                         kernel_med_rel_err=float(np.median(rel)),
                         s_min=float(s.min()), s_max=float(s.max()),
                         rule=dict(eps=info["eps"], ratio=info["ratio"],
                                   n_leg=info["n_leg"], n_jacobi=info["n_jacobi"])))
    return rows


def main():
    rep = {}
    print("[1] spectral density <-> Laplace transform 1/(p^a+lam)")
    rep["spectral"] = check_spectral()
    for k, v in rep["spectral"].items():
        print(f"    {k}: max rel err {v:.2e}")

    print("[2] L1 vs exact step response (f=1, alpha=0.5, T=1)")
    rep["l1_convergence"] = check_l1_convergence()
    for r in rep["l1_convergence"]["table"]:
        print(f"    N={r['N']:5d}  rel_err={r['rel_err']:.3e}")
    print(f"    observed orders: "
          f"{['%.2f' % o for o in rep['l1_convergence']['observed_orders']]}")

    print("[3] recursion vs FFT convolution (LTI property)")
    rep["lti_max_rel_err"] = check_lti()
    print(f"    max rel diff {rep['lti_max_rel_err']:.2e}")

    print("[4] discrete L1 kernel g[n] vs exact K(n tau)")
    rep["discrete_kernel"] = check_discrete_kernel()
    print(f"    max rel err (t > 10 tau): "
          f"{rep['discrete_kernel']['max_rel_err_for_t_gt_10tau']:.2e}")

    print("[5] Jiang rule for t^-beta on [tau, T] = [0.01, 100]")
    rep["jiang_power"] = check_jiang()
    for r in rep["jiang_power"]:
        print(f"    a={r['alpha']} eps={r['eps']:.0e}  n_exp={r['n_exp']:4d}  "
              f"err={r['max_rel_err']:.2e}  s in [{r['s_min']:.2e},{r['s_max']:.2e}]")

    print("[6] Jiang nodes (<=64) + LS weights vs discrete kernel")
    rep["soe_fit"] = check_soe_fit()
    for r in rep["soe_fit"]:
        print(f"    a={r['alpha']}  n_exp={r['n_exp']}  "
              f"power_err={r['power_rule_err']:.2e}  "
              f"kernel_max={r['kernel_max_rel_err']:.2e}  "
              f"kernel_med={r['kernel_med_rel_err']:.2e}")

    with open(os.path.join(RESULTS, "validation.json"), "w") as fh:
        json.dump(rep, fh, indent=2)
    print(f"\nwrote {os.path.join(RESULTS, 'validation.json')}")


if __name__ == "__main__":
    main()
