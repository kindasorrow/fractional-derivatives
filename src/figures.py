r"""All figures.  Run:  python -m src.figures  [--only name1,name2]

Figures whose inputs are missing are skipped with a note, so this module can be
run before the benchmarks finish.  No torch import here on purpose: under
Anaconda, mixing torch's libiomp with Anaconda's MKL in one process trips
OMP error #15 (a pip venv with OpenBLAS numpy is unaffected).
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from . import exact, jiang
from .config import (ALPHAS_HELDOUT, ALPHAS_TRAIN, DATA, LAM, L_LONG,
                     L_SCALING, L_TRAIN, N_STATES, RESULTS, TAU)
from .l1 import discrete_kernel
from .bands import (ABLATIONS, MAIN3, S_HI, S_LO_TRAIN, T_LONG, T_TRAIN,
                    jiang_nodes)
from .style import (GRID, INK, INK2, INK3, MODEL_STYLE, REFERENCE, SLOT,
                    despine, finish, plot_model, st, use_style)

use_style()
F = lambda n: os.path.join(RESULTS, n)
dpath = lambda n: os.path.join(DATA, n)
ABL = ABLATIONS


def _load(name):
    p = F(name)
    if not os.path.exists(p):
        print(f"    (skip: {name} missing)")
        return None
    with open(p) as fh:
        return json.load(fh)


def _slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y) & (y > 0)
    if m.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(x[m]), np.log(y[m]), 1)[0])


# ===========================================================================
# 1.  the operator, its kernel, and Jiang's rule   (no training needed)
# ===========================================================================
def fig_kernel_and_soe():
    fig, ax = plt.subplots(2, 2, figsize=(9.8, 7.2))

    # (a) the solution kernel and its two power laws -----------------------
    a0 = ax[0, 0]
    t = np.exp(np.linspace(np.log(TAU), np.log(300.0), 300))
    for i, al in enumerate((0.3, 0.5, 0.8)):
        K = exact.kernel(t, al, LAM)
        a0.loglog(t, K, color=SLOT[i], lw=1.8, label=rf"$\alpha$ = {al}")
        g = discrete_kernel(al, LAM, TAU, L_LONG)
        n = np.unique(np.round(np.exp(np.linspace(0, np.log(L_LONG), 22))).astype(int))
        a0.loglog(n * TAU, g[n - 1], ls="none", marker="o", ms=3.5,
                  mfc="none", mec=SLOT[i], mew=0.9)
    sm, lg = exact.kernel_asymptotes(t, 0.5, LAM)
    a0.loglog(t[t < 0.8], sm[t < 0.8], **REFERENCE)
    a0.loglog(t[t > 1.2], lg[t > 1.2], **REFERENCE)
    a0.plot([], [], label=r"$\alpha$=0.5 asymptotes:  "
                         r"$t^{\alpha-1}/\Gamma(\alpha)$,  "
                         r"$\frac{\alpha}{\Gamma(1-\alpha)\lambda^{2}}t^{-1-\alpha}$", **REFERENCE)
    a0.set(xlabel=r"$t$", ylabel=r"$K(t)$", ylim=(1e-7, 80),
           title=r"(a) Memory kernel $K(t)=t^{\alpha-1}E_{\alpha,\alpha}(-\lambda t^{\alpha})$")
    a0.legend(title=r"lines: exact,   circles: L1 scheme",
              title_fontsize=8, loc="lower left", fontsize=7.5)
    despine(a0)

    # (b) a power law is a sum of exponentials ----------------------------
    a1 = ax[0, 1]
    al, beta = 0.5, 0.5
    s, w = jiang.soe_power(beta, TAU, T_LONG, eps=1e-6, n_jacobi=4, n_leg=4,
                           ratio=8.0)
    tt = np.exp(np.linspace(np.log(TAU), np.log(T_LONG), 300))
    for sj, wj in zip(s, w):
        a1.loglog(tt, wj * np.exp(-sj * tt), color=INK3, lw=0.6, alpha=0.7)
    a1.loglog(tt, jiang.soe_eval(s, w, tt), color=SLOT[2], lw=2.6,
              label=rf"$\sum_{{j=1}}^{{{len(s)}}} w_j e^{{-s_j t}}$")
    a1.loglog(tt, tt ** (-beta), color=INK, lw=1.2, ls=(0, (4, 2)),
              label=rf"$t^{{\alpha-1}}$ exact ($\alpha$={al})")
    a1.set(ylim=(1e-3, 3e2), xlabel=r"$t$", ylabel="kernel",
           title=rf"(b) Jiang's rule: {len(s)} exponentials tile a power law")
    a1.legend(loc="upper right")
    despine(a1)

    # (c) where the nodes sit, against the spectral density ---------------
    a2 = ax[1, 0]
    sg = np.exp(np.linspace(np.log(1e-5), np.log(1e4), 500))
    for i, al in enumerate((0.3, 0.5, 0.8)):
        mu = exact.spectral_density(sg, al, LAM)
        a2.loglog(sg, sg * mu, color=SLOT[i], lw=1.6, label=rf"$\alpha$ = {al}")
    sj, _ = jiang_nodes(T_LONG, N_STATES)
    a2.plot(sj, np.full_like(sj, 6.0e-5), ls="none", marker="|", ms=10,
            mec=INK, mew=1.0, label=f"Jiang nodes ({len(sj)})")
    for x, lab in ((1.0 / T_LONG, r"$1/T_{\rm long}$"),
                   (1.0 / T_TRAIN, r"$1/T_{\rm train}$"),
                   (1.0 / TAU, r"$1/\tau$")):
        a2.axvline(x, color=INK3, lw=0.8, ls=":")
        a2.text(x, 3.0, lab, color=INK2, fontsize=8, ha="center")
    a2.set(xlabel=r"relaxation rate $s$",
           ylabel=r"$s\,\mu(s)$   (spectral mass per unit $\ln s$)",
           ylim=(3e-5, 8.0),
           title="(c) Spectral density of the operator and the Jiang nodes")
    a2.legend(loc="upper left", fontsize=7.5)
    despine(a2)

    # (d) accuracy vs number of exponentials ------------------------------
    a3 = ax[1, 1]
    for i, al in enumerate((0.3, 0.5, 0.8)):
        beta = 1.0 - al
        ns, errs = [], []
        for eps in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 1e-12):
            for ratio, nl in ((8.0, 2), (8.0, 3), (8.0, 4), (4.0, 4),
                              (4.0, 6), (2.0, 6), (2.0, 8)):
                try:
                    s, w = jiang.soe_power(beta, TAU, T_LONG, eps=eps,
                                           n_jacobi=4, n_leg=nl, ratio=ratio)
                except ValueError:
                    continue
                e, _, _ = jiang.soe_power_error(beta, s, w, TAU, T_LONG)
                ns.append(len(s))
                errs.append(e)
        ns, errs = np.array(ns), np.array(errs)
        o = np.argsort(ns)
        a3.semilogy(ns[o], np.minimum.accumulate(errs[o]), color=SLOT[i],
                    lw=1.8, label=rf"$\alpha$ = {al}")
    a3.axvline(N_STATES, color=INK3, lw=0.8, ls=":")
    a3.text(N_STATES + 2, 2e-2, f"budget used\nhere: N={N_STATES}",
            color=INK2, fontsize=8)
    a3.set(xlabel=r"number of exponentials $N_{\exp}$",
           ylabel=r"max relative error on $[\tau, T]$", xlim=(0, 140),
           title=rf"(d) Cost of the SOE rule, $T/\tau = {int(round(T_LONG / TAU))}$")
    a3.legend()
    despine(a3)

    fig.tight_layout()
    finish(fig, F("fig01_kernel_and_soe.png"))


# ===========================================================================
# 2.  the data
# ===========================================================================
def fig_data_examples():
    d = np.load(dpath(f"test_L{L_TRAIN}.npz"), allow_pickle=True)
    kinds = json.loads(str(d["meta"]))["kinds"]
    want = ["white", "ou", "smooth", "steps", "bursts", "ramp_noise"]
    fig, ax = plt.subplots(2, 3, figsize=(10.4, 4.8), sharex=True)
    t = np.arange(1, L_TRAIN + 1) * TAU
    for k, kn in enumerate(want):
        a = ax[k // 3, k % 3]
        ki = kinds.index(kn)
        idx = np.where(d["kind_id"] == ki)[0]
        a.plot(t, d["f"][idx[0]], color=INK3, lw=0.7, label=r"$f$ (input)")
        for j, al in enumerate((0.2, 0.5, 0.9)):
            m = idx[np.abs(d["alpha"][idx] - al) < 1e-6]
            if len(m):
                a.plot(t, d["y"][m[0]], color=SLOT[j], lw=1.5,
                       label=rf"$y$, $\alpha$={al}")
        a.set_title(kn, fontsize=9)
        despine(a)
        if k == 0:
            a.legend(ncol=2, fontsize=7, loc="upper right")
        if k >= 3:
            a.set_xlabel(r"$t$")
    fig.suptitle(r"Forcing families and their L1 solutions "
                 r"($\lambda=1$, $\tau=0.01$; $f$ has unit RMS)", fontsize=10)
    fig.tight_layout()
    finish(fig, F("fig02_data_examples.png"))


# ===========================================================================
# 3.  training curves
# ===========================================================================
def fig_training():
    import glob
    runs = {}
    for p in glob.glob(F("train_*.json")):
        d = json.load(open(p))
        if "history" in d and len(d["history"]) > 3:
            runs[d["name"]] = d
    if not runs:
        print("    (skip: no training logs)")
        return
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for group, a, ttl in ((MAIN3, ax[0], "(a) the three requested models"),
                          (ABL, ax[1], "(b) variants and ablations")):
        for nm in group:
            if nm not in runs:
                continue
            h = runs[nm]["history"]
            c, ls, mk, lab = st(nm)
            a.semilogy([r["step"] for r in h], [r["val_rel_l2_med"] for r in h],
                       color=c, ls=ls, lw=1.6, label=lab)
        a.set(xlabel="optimisation step",
              ylabel=r"validation median $\|\hat y-y\|_2/\|y\|_2$", title=ttl)
        a.legend(fontsize=7.5, ncol=1)
        despine(a)
    fig.suptitle(r"Training at $L=1000$ (identical data, steps and loss for every model)",
                 fontsize=10)
    fig.tight_layout()
    finish(fig, F("fig03_training.png"))


# ===========================================================================
# 4.  MAIN EXPERIMENT: accuracy vs sequence length
# ===========================================================================
def fig_accuracy_vs_length():
    b = _load("bench.json")
    if b is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10.0, 4.2), sharey=True)
    soe_ls = b.get("soe_ls", {})

    for group, a, ttl, loc, ncol in (
            (MAIN3, ax[0], "(a) the three requested models", "center left", 1),
            (ABL, ax[1], "(b) variants and ablations", "lower left", 2)):
        for nm in group:
            m = b["models"].get(nm)
            if m is None:
                continue
            L = [r["L"] for r in m["scaling"] if "med" in r["acc"]]
            e = [r["acc"]["med"] for r in m["scaling"] if "med" in r["acc"]]
            if not L:
                continue
            plot_model(a, nm, L, e, ms=5)
        if soe_ls:
            Ls = sorted(int(k) for k in soe_ls)
            a.plot(Ls, [soe_ls[str(k)]["med"] for k in Ls], **REFERENCE)
            if ncol == 1:          # label the reference once, in panel (a)
                a.text(Ls[0] * 1.05, soe_ls[str(Ls[0])]["med"] * 2.4,
                       "Jiang nodes + LS weights (no training)", color=INK2,
                       fontsize=7.5)
        a.set(xscale="log", yscale="log", xlabel="sequence length $L$",
              title=ttl, xlim=(880, 11500))
        a.axvline(L_TRAIN, color=INK2, lw=1.0)
        a.legend(fontsize=7.0 if ncol > 1 else 7.5, loc=loc, ncol=ncol,
                 columnspacing=1.0, handlelength=2.2,
                 bbox_to_anchor=(0.0, 0.08) if ncol > 1 else None)
        despine(a)
    ax[0].set_ylabel(r"median relative $L_2$ error")
    ax[0].text(L_TRAIN * 1.12, 2.2,
               r"trained here $\rightarrow$ extrapolation", color=INK2,
               fontsize=8)
    fig.suptitle("Trained on $L=1000$, evaluated up to $L=10000$ "
                 "(same $\\tau$, so $T$ grows from 10 to 100)", fontsize=10)
    fig.tight_layout()
    finish(fig, F("fig04_accuracy_vs_length.png"))


# ===========================================================================
# 5.  time and memory
# ===========================================================================
def fig_time_memory():
    b = _load("bench.json")
    if b is None:
        return
    fig, ax = plt.subplots(1, 3, figsize=(12.4, 3.9))
    names = [n for n in MAIN3 + ABL if n in b["models"]]

    for nm in names:
        sc = b["models"][nm]["scaling"]
        L = [r["L"] for r in sc if r["speed"]["ok"]]
        t = [r["speed"]["time_ms"] for r in sc if r["speed"]["ok"]]
        if not L:
            continue
        kw = dict(ms=5) if nm in MAIN3 else dict(ms=3.5, alpha=0.55, lw=1.1)
        plot_model(ax[0], nm, L, t, **kw)
        if nm in MAIN3:
            ax[0].annotate(rf"$\propto L^{{{_slope(L, t):.2f}}}$",
                           (L[-1], t[-1]), xytext=(5, -2),
                           textcoords="offset points", color=st(nm)[0], fontsize=8)
    ax[0].set(xscale="log", yscale="log", xlabel="sequence length $L$",
              ylabel="forward time per sequence [ms]",
              title=f"(a) inference latency ({b['device']}, batch 1)")
    ax[0].legend(fontsize=7, ncol=1, loc="upper left")
    despine(ax[0])

    for nm in names:
        sc = b["models"][nm]["scaling"]
        L = [r["L"] for r in sc if r["speed"]["ok"]]
        m = [r["speed"]["peak_MiB"] for r in sc if r["speed"]["ok"]]
        if not L:
            continue
        kw = dict(ms=5) if nm in MAIN3 else dict(ms=3.5, alpha=0.55, lw=1.1)
        plot_model(ax[1], nm, L, m, **kw)
        if nm in MAIN3:
            ax[1].annotate(rf"$\propto L^{{{_slope(L, m):.2f}}}$",
                           (L[-1], m[-1]), xytext=(5, -2),
                           textcoords="offset points", color=st(nm)[0], fontsize=8)
    ax[1].set(xscale="log", yscale="log", xlabel="sequence length $L$",
              ylabel="peak allocated memory [MiB]",
              title="(b) inference memory (batch 1)")
    despine(ax[1])

    cl = b.get("classical", [])
    if cl:
        L = [r["L"] for r in cl]
        for key, lab, col in (("l1_direct_ms", r"L1 recursion, $O(L^2)$", SLOT[7]),
                              ("l1_fft_ms", r"L1 as FFT convolution", SLOT[6]),
                              ("soe_ls_ms", rf"Jiang SOE recursion, $N$={N_STATES}", SLOT[2])):
            y = [r[key] for r in cl]
            ax[2].loglog(L, y, color=col, lw=1.8, marker="o", ms=5,
                         label=rf"{lab}  ($\propto L^{{{_slope(L, y):.2f}}}$)")
        ax[2].set(xlabel="sequence length $L$", ylabel="solve time [ms]",
                  title="(c) classical solvers (CPU, numpy/scipy)")
        ax[2].legend(fontsize=7.5)
        despine(ax[2])
    fig.tight_layout()
    finish(fig, F("fig05_time_memory.png"))


# ===========================================================================
# 6.  where the error lives in time
# ===========================================================================
def fig_error_vs_time():
    b = _load("bench.json")
    if b is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(9.8, 4.0))
    for nm in MAIN3 + ABL:
        m = b["models"].get(nm, {}).get("long_test")
        if not m or "rms_err" not in m:
            continue
        e = np.asarray(m["rms_err"])
        t = (np.arange(len(e)) * m["stride"] + 1) * TAU
        c, ls, _, lab = st(nm)
        kw = dict(lw=1.8) if nm in MAIN3 else dict(lw=1.0, alpha=0.5)
        ax[0].loglog(t, e, color=c, ls=ls, label=lab, **kw)
        if nm in MAIN3:
            ry = np.asarray(m["rms_y"])
            ax[1].loglog(t, e / ry, color=c, ls=ls, label=lab, lw=1.8)
    m0 = next((b["models"][n]["long_test"] for n in MAIN3 + ABL
               if "rms_y" in b["models"].get(n, {}).get("long_test", {})), None)
    if m0 is None:
        print("    (skip: no long_test data)")
        plt.close(fig)
        return
    ry = np.asarray(m0["rms_y"])
    t = (np.arange(len(ry)) * m0["stride"] + 1) * TAU
    ax[0].loglog(t, ry, label=r"RMS of the target $y$", **REFERENCE)
    for a in ax:
        a.axvline(T_TRAIN, color=INK2, lw=1.0)
        a.text(T_TRAIN * 1.1, a.get_ylim()[0] * 2, "training horizon",
               rotation=90, color=INK2, fontsize=7.5, va="bottom")
        a.set_xlabel(r"$t$")
        despine(a)
    ax[0].set(ylabel="RMS error over the test set",
              title="(a) absolute error vs time, $L=10000$")
    ax[1].set(ylabel="RMS error / RMS of $y$",
              title="(b) the same, normalised")
    ax[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    finish(fig, F("fig06_error_vs_time.png"))


# ===========================================================================
# 7.  TASK 5: learned rates vs Jiang nodes
# ===========================================================================
def fig_rates_vs_jiang():
    an = _load("analysis.json")
    if an is None:
        return
    sj = np.asarray(an["jiang"]["long"]["s"])
    rows = [("s4d_lin_log", "S4D-linear, log init"),
            ("s4d_lin_linear", "S4D-linear, linear init"),
            ("s4d_lin_narrow", "S4D-linear, 1-scale init"),
            ("s4d_lin_softplus", "S4D-linear, softplus"),
            ("s4d_deep", "S4D deep (all layers)"),
            ("s4d_lin_log_longtrain", "S4D-linear, trained at $L$=10000"),
            ("s4d_lin_narrow_longtrain", "S4D-lin 1-scale, trained at $L$=10000")]
    rows = [r for r in rows if r[0] in an["models"]]

    fig = plt.figure(figsize=(11.6, 7.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.42, wspace=0.28)

    # (a) strip plot ------------------------------------------------------
    a0 = fig.add_subplot(gs[0, :])
    yl = []
    a0.plot(sj, np.full_like(sj, 0.0), ls="none", marker="|", ms=16, mew=1.3,
            color=INK)
    yl.append(f"Jiang nodes ($N$={len(sj)})")
    for k, (nm, lab) in enumerate(rows, start=1):
        d = an["models"][nm]
        s = np.asarray(d["rates"])
        w = np.asarray(d["importance"])
        w = w / w.max()
        c = st(nm)[0]
        a0.vlines(s, -k - 0.3, -k + 0.3, colors=c, alpha=0.65,
                  linewidths=0.35 + 1.9 * w)
        a0.plot(s.min(), -k, marker="<", ms=5, color=c, ls="none")
        yl.append(lab)
    for x, lab in ((1.0 / T_LONG, r"$1/T_{\rm long}$"),
                   (1.0 / T_TRAIN, r"$1/T_{\rm train}$"),
                   (1.0 / TAU, r"$1/\tau$")):
        a0.axvline(x, color=INK3, lw=0.9, ls=":")
        a0.text(x, 0.75, lab, color=INK2, fontsize=8, ha="center")
    a0.axvspan(S_LO_TRAIN, S_HI, color=GRID, alpha=0.7, lw=0, zorder=0)
    a0.text(np.sqrt(S_LO_TRAIN * S_HI), -len(rows) - 0.75,
            "shaded: band the SSMs were initialised in", color=INK2,
            fontsize=8, ha="center")
    a0.set(xscale="log", yticks=[-k for k in range(len(yl))],
           yticklabels=yl, xlim=(1e-5, 1e4),
           ylim=(-len(rows) - 1.1, 1.1), xlabel=r"relaxation rate $s$  [1/time]",
           title="(a) Learned relaxation rates (tick width $\\propto$ mode importance) "
                 "against Jiang's nodes")
    a0.grid(axis="y", visible=False)
    despine(a0)

    # (b) nodes per decade ------------------------------------------------
    a1 = fig.add_subplot(gs[1, 0])
    # shares, not counts: the deep SSM has 8192 poles and Jiang has 64, so only
    # the *shape* of the distribution over decades is comparable
    edges = np.arange(-5, 5)
    width = 0.8 / (1 + len(rows))
    ctr = edges[:-1] + 0.5
    cj, _ = np.histogram(np.log10(sj), bins=edges)
    a1.bar(ctr - 0.4 + 0.5 * width, cj / cj.sum(), width=width, color=INK,
           label="Jiang", zorder=3)
    for k, (nm, lab) in enumerate(rows, start=1):
        d = an["models"][nm]
        s, w = np.asarray(d["rates"]), np.asarray(d["importance"])
        keep = w > 0.01 * w.max()
        h, _ = np.histogram(np.log10(s[keep]), bins=edges)
        a1.bar(ctr - 0.4 + (k + 0.5) * width, h / max(1, h.sum()), width=width,
               color=st(nm)[0], label=st(nm)[3], zorder=3)
    a1.set(xlabel=r"$\log_{10} s$", ylabel="share of modes in the decade",
           title="(b) distribution of rates over decades")
    a1.legend(fontsize=6.5)
    despine(a1)

    # (c) QQ plot ----------------------------------------------------------
    a2 = fig.add_subplot(gs[1, 1])
    lo, hi = -5.0, 4.0
    a2.plot([lo, hi], [lo, hi], color=INK3, lw=1.0, ls=":")
    for nm, lab in rows:
        d = an["models"][nm]
        q = d.get("qq_sig_vs_jiang") or d["qq_vs_jiang"]
        a2.plot(q["jiang"], q["learned"], color=st(nm)[0], lw=1.6,
                marker=st(nm)[2], ms=4, label=st(nm)[3])
    a2.set(xlabel=r"Jiang $\log_{10}s$ quantiles",
           ylabel=r"learned $\log_{10}s$ quantiles",
           title="(c) quantile-quantile vs Jiang")
    a2.legend(fontsize=6.0, loc="lower right")
    despine(a2)

    # (d) weights vs rate --------------------------------------------------
    a3 = fig.add_subplot(gs[1, 2])
    g = discrete_kernel(0.5, LAM, TAU, L_LONG)
    c_ls = jiang.fit_weights_to_kernel(sj, g, TAU)
    a3.loglog(sj, np.abs(c_ls), color=INK, lw=0, marker="o", ms=4,
              label=r"Jiang nodes, LS weights $|c_j|$")
    for nm in ("soe_jiang", "s4d_lin_log"):
        d = an["models"].get(nm)
        if d is None or d.get("kind") != "single_layer":
            continue
        ai = int(np.argmin(np.abs(np.asarray(d["alphas"]) - 0.5)))
        a3.loglog(np.asarray(d["rates"]), np.abs(np.asarray(d["coeffs"])[ai]),
                  color=st(nm)[0], lw=0, marker=st(nm)[2], ms=4, alpha=0.8,
                  label=st(nm)[3])
    a3.set(xlabel=r"rate $s_j$", ylabel=r"$|c_j|$",
           title=r"(d) readout weights, $\alpha=0.5$")
    a3.legend(fontsize=6.5)
    despine(a3)

    fig.suptitle("Do the rates a network learns agree with Jiang's quadrature?",
                 fontsize=11)
    finish(fig, F("fig07_rates_vs_jiang.png"))


# ===========================================================================
# 8.  how far the rates moved from their initialisation
# ===========================================================================
def fig_rate_movement():
    an = _load("analysis.json")
    if an is None:
        return
    rows = [n for n in ("s4d_lin_log", "s4d_lin_narrow", "s4d_deep",
                        "s4d_lin_log_longtrain", "s4d_lin_narrow_longtrain")
            if n in an["models"]]
    fig, ax = plt.subplots(1, len(rows), figsize=(3.1 * len(rows), 3.5),
                           sharex=True, sharey=True)
    ax = np.atleast_1d(ax)
    sj = np.asarray(an["jiang"]["long"]["s"])
    for a, nm in zip(ax, rows):
        d = an["models"][nm]
        s0 = np.asarray(d["init_rates"])
        s1 = np.asarray(d["rates"])
        w = np.asarray(d["importance"])
        w = w / w.max()
        a.plot([1e-5, 1e4], [1e-5, 1e4], color=INK3, lw=1.0, ls=":")
        a.scatter(s0, s1, s=4 + 60 * w, facecolor=st(nm)[0], alpha=0.55,
                  edgecolor="none")
        for x in (sj.min(), sj.max()):
            a.axhline(x, color=INK, lw=0.8, ls=(0, (4, 2)))
        a.set(xscale="log", yscale="log", xlim=(1e-5, 1e4), ylim=(1e-5, 1e4),
              xlabel=r"initial rate $s_j^{(0)}$", title=st(nm)[3], aspect="equal")
        despine(a)
    ax[0].set_ylabel(r"learned rate $s_j$")
    ax[0].text(2e-5, 1.0e-3, "dashed: edges of\nthe Jiang band",
               color=INK2, fontsize=7, va="bottom")
    ax[0].text(3e-3, 1.1e-4, "dotted: unchanged", color=INK3, fontsize=7,
               rotation=45, va="bottom")
    fig.suptitle("Rate movement during training (dot size $\\propto$ importance)",
                 fontsize=10)
    fig.tight_layout()
    finish(fig, F("fig08_rate_movement.png"))


# ===========================================================================
# 9.  the convolution kernel each model actually realises
# ===========================================================================
def fig_implied_kernel():
    an = _load("analysis.json")
    if an is None:
        return
    als = ["0.3", "0.7"]
    als = [a for a in als if a in an["kernels"]]
    fig, ax = plt.subplots(2, len(als), figsize=(5.0 * len(als), 6.2),
                           sharex=True)
    ax = ax.reshape(2, len(als))
    for j, akey in enumerate(als):
        e = an["kernels"][akey]
        n = np.asarray(e["n"])
        t = n * TAU
        true = np.asarray(e["true"])
        ax[0, j].loglog(t, true, color=INK, lw=2.2, label="true L1 kernel")
        ax[0, j].loglog(t, np.abs(np.asarray(e["soe_ls"])),
                        label="Jiang + LS weights (no training)", **REFERENCE)
        ax[1, j].loglog(t, np.abs(np.asarray(e["soe_ls"]) - true) / true,
                        **REFERENCE)
        for nm in ("soe_jiang", "s4d_lin_log", "s4d_lin_narrow"):
            if nm not in e:
                continue
            k = np.asarray(e[nm])
            c, ls, _, lab = st(nm)
            ax[0, j].loglog(t, np.abs(k), color=c, ls=ls, lw=1.5, label=lab)
            ax[1, j].loglog(t, np.abs(k - true) / true, color=c, ls=ls, lw=1.5)
        for a in ax[:, j]:
            a.axvline(T_TRAIN, color=INK2, lw=1.0)
            despine(a)
        ax[0, j].set_title(rf"$\alpha$ = {akey}")
        ax[1, j].set_xlabel(r"$t$")
    ax[0, 0].set_ylabel(r"$|$kernel$|$")
    ax[1, 0].set_ylabel("relative error of the kernel")
    ax[0, 0].legend(fontsize=7)
    ax[0, 0].text(T_TRAIN * 1.15, true.max() * 0.2, "training\nhorizon",
                  color=INK2, fontsize=7)
    fig.suptitle("The impulse response each learned model realises", fontsize=10)
    fig.tight_layout()
    finish(fig, F("fig09_implied_kernel.png"))


# ===========================================================================
# 10.  per-alpha and held-out alpha
# ===========================================================================
def fig_alpha_breakdown():
    b = _load("bench.json")
    if b is None:
        return
    fig, ax = plt.subplots(1, 3, figsize=(12.0, 3.8))
    for k, L in enumerate((L_TRAIN, L_LONG)):
        for nm in MAIN3:
            m = b["models"].get(nm)
            if m is None:
                continue
            rec = [r for r in m["scaling"] if r["L"] == L]
            if not rec or "per_alpha" not in rec[0]["acc"]:
                continue
            pa = {float(a): v for a, v in rec[0]["acc"]["per_alpha"].items()}
            xs = sorted(pa)
            plot_model(ax[k], nm, xs, [pa[a] for a in xs])
        ax[k].set(yscale="log", xlabel=r"fractional order $\alpha$",
                  title=rf"({'ab'[k]}) $L$ = {L}")
        despine(ax[k])
    ax[0].set_ylabel(r"median relative $L_2$ error")
    ax[0].legend(fontsize=7.5)

    for nm in MAIN3:
        ho = b["models"].get(nm, {}).get("heldout_alpha")
        if not ho or "error" in ho:
            continue
        hh = {float(a): v for a, v in ho.items() if a != "all_med"}
        xs = sorted(hh)
        plot_model(ax[2], nm, xs, [hh[a] for a in xs])
    ax[2].set(yscale="log", xlabel=r"unseen $\alpha$",
              title=r"(c) held-out orders, $L$ = 10000")
    ax[2].legend(fontsize=7.5)
    despine(ax[2])
    fig.tight_layout()
    finish(fig, F("fig10_alpha_breakdown.png"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    reg = {k[4:]: v for k, v in globals().items() if k.startswith("fig_")}
    names = args.only.split(",") if args.only else list(reg)
    for nm in names:
        print(f"[fig] {nm}")
        reg[nm]()
