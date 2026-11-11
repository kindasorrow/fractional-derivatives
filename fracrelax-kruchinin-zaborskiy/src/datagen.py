r"""Dataset generation for D^a y = -lam y + f, solved with the L1 scheme.

Design notes
------------
* tau is FIXED (config.TAU).  A longer sequence therefore means a longer time
  horizon T = L*tau and a longer memory, which is the point of the
  "train short / test long" experiment.
* y(0) = 0, so f -> y is exactly linear & time-invariant and the L1 solution
  equals a convolution with the scheme's own impulse response.  Generation uses
  that (FFT) after `src.validate` confirmed it to 5e-16.
* Every forcing family is defined with *length-independent* statistics (rates
  and frequencies in physical units), so the only thing that changes between a
  short and a long sequence is how much of the memory kernel is exercised.
* f is normalised to unit RMS per trajectory; y keeps its natural scale.

Run:  python -m src.datagen            (all datasets)
      python -m src.datagen --demo     (only the small committed demo file)
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from .config import (ALPHAS_HELDOUT, ALPHAS_TRAIN, DATA, LAM, L_LONG,
                     L_SCALING, L_TRAIN, SEED, TAU)
from .l1 import discrete_kernel, l1_solve_fft

KINDS = ("white", "ou", "smooth", "steps", "bursts", "chirp", "ramp_noise", "mix")


# ---------------------------------------------------------------------------
# forcing families
# ---------------------------------------------------------------------------
def _draw_one(kind: str, L: int, tau: float, rng: np.random.Generator) -> np.ndarray:
    t = (np.arange(1, L + 1)) * tau
    if kind == "white":
        return rng.standard_normal(L)
    if kind == "ou":
        theta = np.exp(rng.uniform(np.log(0.02), np.log(5.0)))  # corr. time [s]
        phi = np.exp(-tau / theta)
        xi = rng.standard_normal(L) * np.sqrt(1.0 - phi ** 2)
        x = np.empty(L)
        x[0] = rng.standard_normal()
        for n in range(1, L):           # cheap enough; L <= 1e4
            x[n] = phi * x[n - 1] + xi[n]
        return x
    if kind == "smooth":
        K = rng.integers(2, 7)
        w = np.exp(rng.uniform(np.log(0.05), np.log(30.0), K))
        a = np.abs(rng.standard_normal(K)) + 0.2
        p = rng.uniform(0, 2 * np.pi, K)
        return (a[:, None] * np.sin(w[:, None] * t[None, :] + p[:, None])).sum(0)
    if kind == "steps":
        rate = np.exp(rng.uniform(np.log(0.05), np.log(5.0)))   # jumps per sec
        jump = rng.random(L) < rate * tau
        idx = np.cumsum(jump)
        lev = rng.standard_normal(idx.max() + 1)
        return lev[idx]
    if kind == "bursts":
        dens = np.exp(rng.uniform(np.log(5e-4), np.log(2e-2)))  # per step
        x = np.zeros(L)
        m = rng.random(L) < dens
        x[m] = rng.standard_normal(m.sum())
        if not m.any():
            x[rng.integers(L)] = 1.0
        return x
    if kind == "chirp":
        f0 = np.exp(rng.uniform(np.log(0.02), np.log(2.0)))
        k = np.exp(rng.uniform(np.log(1e-3), np.log(0.5)))
        return np.sin(2 * np.pi * (f0 * t + 0.5 * k * t * t))
    if kind == "ramp_noise":
        # slow polynomial trend + white noise: excites the slow SSM modes
        p = rng.uniform(0.5, 2.0)
        tr = (t / t[-1]) ** p * rng.choice([-1.0, 1.0])
        return 3.0 * tr + 0.5 * rng.standard_normal(L)
    if kind == "mix":
        k1, k2 = rng.choice(KINDS[:-1], 2, replace=False)
        a = rng.uniform(0.3, 0.7)
        x1, x2 = _draw_one(k1, L, tau, rng), _draw_one(k2, L, tau, rng)
        x1 = x1 / (x1.std() + 1e-12)
        x2 = x2 / (x2.std() + 1e-12)
        return a * x1 + (1 - a) * x2
    raise ValueError(kind)


def draw_forcing(n: int, L: int, tau: float, rng: np.random.Generator):
    """Return (f, kind_id) with f normalised to unit RMS per row."""
    f = np.empty((n, L))
    kid = np.empty(n, dtype=np.int16)
    for i in range(n):
        j = int(rng.integers(len(KINDS)))
        x = _draw_one(KINDS[j], L, tau, rng)
        x = x - 0.0  # keep the mean: a DC component is physically meaningful
        f[i] = x / (np.sqrt(np.mean(x ** 2)) + 1e-12)
        kid[i] = j
    return f, kid


# ---------------------------------------------------------------------------
# dataset builders
# ---------------------------------------------------------------------------
def build_fixed_length(path: str, alphas, n_per_alpha: int, L: int,
                       seed: int, tau: float = TAU, lam: float = LAM,
                       chunk: int = 64):
    rng = np.random.default_rng(seed)
    alphas = np.asarray(alphas, dtype=np.float64)
    n = len(alphas) * n_per_alpha
    F = np.empty((n, L), dtype=np.float32)
    Y = np.empty((n, L), dtype=np.float32)
    A = np.empty(n, dtype=np.float32)
    AID = np.empty(n, dtype=np.int16)
    KID = np.empty(n, dtype=np.int16)
    t0 = time.time()
    p = 0
    for ai, a in enumerate(alphas):
        g = discrete_kernel(a, lam, tau, L)
        for start in range(0, n_per_alpha, chunk):
            m = min(chunk, n_per_alpha - start)
            f, kid = draw_forcing(m, L, tau, rng)
            y = l1_solve_fft(f, a, lam, tau, kernel=g)
            F[p:p + m] = f
            Y[p:p + m] = y
            A[p:p + m] = a
            AID[p:p + m] = ai
            KID[p:p + m] = kid
            p += m
    meta = dict(tau=tau, lam=lam, L=L, alphas=alphas.tolist(), kinds=list(KINDS),
                seed=seed, n=n, scheme="L1 (Caputo), y0=0",
                generated=time.strftime("%Y-%m-%d %H:%M:%S"))
    np.savez_compressed(path, f=F, y=Y, alpha=A, alpha_id=AID, kind_id=KID,
                        alphas=alphas.astype(np.float32), tau=np.float64(tau),
                        lam=np.float64(lam), meta=json.dumps(meta))
    print(f"  {os.path.basename(path):28s} n={n:5d} L={L:6d} "
          f"({time.time() - t0:5.1f}s, {os.path.getsize(path) / 1e6:.1f} MB)")
    return meta


def build_mixed_length(path: str, alphas, n: int, lmin: int, lmax: int,
                       seed: int, tau: float = TAU, lam: float = LAM):
    """The literal task spec: random lengths in [lmin, lmax].

    Stored zero-padded to lmax with an explicit `length` array.
    """
    rng = np.random.default_rng(seed)
    alphas = np.asarray(alphas, dtype=np.float64)
    F = np.zeros((n, lmax), dtype=np.float32)
    Y = np.zeros((n, lmax), dtype=np.float32)
    A = np.empty(n, dtype=np.float32)
    AID = np.empty(n, dtype=np.int16)
    KID = np.empty(n, dtype=np.int16)
    LEN = np.empty(n, dtype=np.int32)
    t0 = time.time()
    for i in range(n):
        ai = int(rng.integers(len(alphas)))
        L = int(rng.integers(lmin, lmax + 1))
        f, kid = draw_forcing(1, L, tau, rng)
        y = l1_solve_fft(f, alphas[ai], lam, tau)
        F[i, :L] = f[0]
        Y[i, :L] = y[0]
        A[i], AID[i], KID[i], LEN[i] = alphas[ai], ai, kid[0], L
    meta = dict(tau=tau, lam=lam, L_min=lmin, L_max=lmax, alphas=alphas.tolist(),
                kinds=list(KINDS), seed=seed, n=n, padded=True,
                scheme="L1 (Caputo), y0=0",
                generated=time.strftime("%Y-%m-%d %H:%M:%S"))
    np.savez_compressed(path, f=F, y=Y, alpha=A, alpha_id=AID, kind_id=KID,
                        length=LEN, alphas=alphas.astype(np.float32),
                        tau=np.float64(tau), lam=np.float64(lam),
                        meta=json.dumps(meta))
    print(f"  {os.path.basename(path):28s} n={n:5d} L={lmin}..{lmax} "
          f"({time.time() - t0:5.1f}s, {os.path.getsize(path) / 1e6:.1f} MB)")
    return meta


def build_kernels(path: str, alphas, L: int, tau: float = TAU, lam: float = LAM):
    G = np.stack([discrete_kernel(a, lam, tau, L) for a in alphas])
    np.savez_compressed(path, g=G.astype(np.float64),
                        alphas=np.asarray(alphas, dtype=np.float32),
                        tau=np.float64(tau), lam=np.float64(lam))
    print(f"  {os.path.basename(path):28s} impulse responses, L={L}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="only the small demo file")
    args = ap.parse_args()

    print("generating datasets in", DATA)
    if args.demo:
        build_fixed_length(os.path.join(DATA, "demo_small.npz"),
                           ALPHAS_TRAIN, 4, L_TRAIN, SEED + 99)
        return

    # --- main training / validation / test, all at the short length ---------
    build_fixed_length(os.path.join(DATA, f"train_L{L_TRAIN}.npz"),
                       ALPHAS_TRAIN, 400, L_TRAIN, SEED + 1)
    build_fixed_length(os.path.join(DATA, f"val_L{L_TRAIN}.npz"),
                       ALPHAS_TRAIN, 50, L_TRAIN, SEED + 2)
    build_fixed_length(os.path.join(DATA, f"test_L{L_TRAIN}.npz"),
                       ALPHAS_TRAIN, 64, L_TRAIN, SEED + 3)

    # --- the long-sequence test set (main experiment) ----------------------
    build_fixed_length(os.path.join(DATA, f"test_L{L_LONG}.npz"),
                       ALPHAS_TRAIN, 48, L_LONG, SEED + 4)

    # --- a long-sequence TRAINING set, used only for the control run that
    #     asks whether an SSM finds Jiang's band when the data shows it -----
    build_fixed_length(os.path.join(DATA, f"train_L{L_LONG}.npz"),
                       ALPHAS_TRAIN, 100, L_LONG, SEED + 7)
    build_fixed_length(os.path.join(DATA, f"val_L{L_LONG}.npz"),
                       ALPHAS_TRAIN, 16, L_LONG, SEED + 8)

    # --- length sweep for accuracy / time / memory scaling -----------------
    for L in L_SCALING:
        build_fixed_length(os.path.join(DATA, f"scale_L{L}.npz"),
                           ALPHAS_TRAIN, 16, L, SEED + 10 + L)

    # --- unseen fractional orders -----------------------------------------
    build_fixed_length(os.path.join(DATA, f"heldout_alpha_L{L_LONG}.npz"),
                       ALPHAS_HELDOUT, 32, L_LONG, SEED + 5)

    # --- literal spec: random lengths 1000..10000 -------------------------
    build_mixed_length(os.path.join(DATA, "mixed_lengths.npz"),
                       ALPHAS_TRAIN, 240, 1000, 10000, SEED + 6)

    # --- reference impulse responses + small committed demo ---------------
    build_kernels(os.path.join(DATA, f"kernels_L{L_LONG}.npz"),
                  ALPHAS_TRAIN, L_LONG)
    build_fixed_length(os.path.join(DATA, "demo_small.npz"),
                       ALPHAS_TRAIN, 4, L_TRAIN, SEED + 99)
    print("done")


if __name__ == "__main__":
    main()
