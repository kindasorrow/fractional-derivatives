r"""Jiang's sum-of-exponentials (SOE) approximation of a power-law kernel.

Reference: S. Jiang, J. Zhang, Q. Zhang, Z. Zhang, "Fast evaluation of the
Caputo fractional derivative and its applications to fractional diffusion
equations", Commun. Comput. Phys. 21(3), 650-678 (2017).

Idea.  For 0 < b < 1,

    t^{-b} = 1/G(b) \int_0^\infty e^{-s t} s^{b-1} ds .

Discretising the s-integral with a *composite* Gauss rule gives

    t^{-b} ~ sum_{j=1}^{Nexp} w_j e^{-s_j t},    t in [tau, T],

with relative error <= eps and  Nexp = O(log(1/eps) * log(T/tau)).  The rule is

  * [0, s_min]            : Gauss-Jacobi with weight s^{b-1} (absorbs the
                            integrable singularity exactly),
  * [s_min, s_max] cut
    into geometric panels : Gauss-Legendre (the integrand is analytic there,
                            the panels follow the scale of s),
  * [s_max, inf)          : truncated; s_max is picked from the tail bound
                            u^{b-1} e^{-u} / G(b) <= eps, u = s_max * tau.

The resulting nodes s_j are (by construction) spread roughly geometrically over
the band [1/T, log(1/eps)/tau]: a few nodes per octave of relaxation rate.
That node distribution is the object we compare the learned S4D poles against.

Each exponential is one first-order mode:  h_j' = -s_j h_j + f, so the SOE
approximation of the solution operator is literally a diagonal state-space
model with *fixed* poles a_j = exp(-s_j tau).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.signal import lfilter
from scipy.special import gamma as _gamma
from scipy.special import roots_jacobi, roots_legendre


# ---------------------------------------------------------------------------
# the quadrature
# ---------------------------------------------------------------------------
def tail_cut(beta: float, eps: float) -> float:
    """Smallest u with  u^{b-1} e^{-u} / G(b) <= eps  (relative tail bound)."""
    f = lambda u: (u ** (beta - 1.0)) * np.exp(-u) / _gamma(beta) - eps
    lo, hi = 1e-6, 2.0
    while f(hi) > 0:
        hi *= 2.0
        if hi > 1e5:
            break
    return float(brentq(f, lo, hi, xtol=1e-10, rtol=1e-12))


def soe_power(beta: float, tau: float, T: float, eps: float = 1e-6,
              n_jacobi: int = 16, n_leg: int = 8, ratio: float = 2.0,
              c_low: float = 0.5):
    """Nodes/weights with  t^{-beta} ~ sum w_j exp(-s_j t)  on [tau, T].

    Returns (s, w), both sorted by increasing s.
    """
    if not 0.0 < beta < 1.0:
        raise ValueError("beta must lie in (0,1)")
    s_max = tail_cut(beta, eps) / tau
    s_min = c_low / T
    if s_min >= s_max:
        raise ValueError("degenerate band; check tau/T/eps")

    S, W = [], []

    # --- [0, s_min]: Gauss-Jacobi for weight s^{beta-1} ---------------------
    #   \int_0^sig s^{b-1} p(s) ds = (sig/2)^b sum_k om_k p(sig (1+x_k)/2)
    x, om = roots_jacobi(n_jacobi, 0.0, beta - 1.0)
    s = s_min * (1.0 + x) / 2.0
    w = (s_min / 2.0) ** beta * om / _gamma(beta)
    S.append(s)
    W.append(w)

    # --- geometric panels up to s_max: Gauss-Legendre -----------------------
    xg, omg = roots_legendre(n_leg)
    edges = [s_min]
    while edges[-1] < s_max:
        edges.append(edges[-1] * ratio)
    edges[-1] = s_max
    for a, b in zip(edges[:-1], edges[1:]):
        mid, half = 0.5 * (a + b), 0.5 * (b - a)
        s = mid + half * xg
        w = half * omg * s ** (beta - 1.0) / _gamma(beta)
        S.append(s)
        W.append(w)

    s = np.concatenate(S)
    w = np.concatenate(W)
    o = np.argsort(s)
    return s[o], w[o]


def soe_eval(s, w, t):
    """sum_j w_j exp(-s_j t) for an array of t."""
    t = np.atleast_1d(np.asarray(t, dtype=np.float64))
    return np.exp(-np.outer(t, np.asarray(s))) @ np.asarray(w)


def soe_power_error(beta: float, s, w, tau: float, T: float, n_test: int = 400):
    """Max relative error of the SOE approximation of t^{-beta} on [tau, T]."""
    t = np.exp(np.linspace(np.log(tau), np.log(T), n_test))
    approx = soe_eval(s, w, t)
    exact = t ** (-beta)
    return float(np.max(np.abs(approx - exact) / exact)), t, approx / exact - 1.0


def soe_power_budget(beta: float, tau: float, T: float, n_max: int,
                     verbose: bool = False):
    """Best Jiang rule for t^{-beta} that fits in <= n_max exponentials.

    Small grid search over (eps, n_leg, ratio, n_jacobi); returns the config
    with the smallest measured relative error on [tau, T].
    """
    best = None
    for eps in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8):
        for ratio in (2.0, 3.0, 4.0, 6.0, 8.0):
            for n_leg in (2, 3, 4, 5, 6, 8, 10, 12):
                for n_jacobi in (4, 6, 8, 12):
                    try:
                        s, w = soe_power(beta, tau, T, eps, n_jacobi, n_leg, ratio)
                    except ValueError:
                        continue
                    if s.size > n_max:
                        continue
                    err, _, _ = soe_power_error(beta, s, w, tau, T)
                    cand = dict(err=err, eps=eps, ratio=ratio, n_leg=n_leg,
                                n_jacobi=n_jacobi, n_exp=int(s.size))
                    if best is None or err < best["err"]:
                        best = cand
                        best["s"], best["w"] = s, w
    if best is None:
        raise RuntimeError("no admissible rule within the node budget")
    if verbose:
        print({k: v for k, v in best.items() if k not in ("s", "w")})
    return best


def nodes_for_alpha(alpha: float, tau: float, T: float, n_max: int):
    """Jiang nodes for the relaxation kernel of order `alpha`.

    K(t) = t^{a-1} E_{a,a}(-lam t^a) behaves like t^{-(1-a)} for small t and
    like t^{-(1+a)} for large t, so the natural Jiang rule is the one built for
    the exponent beta = 1 - alpha over the same band [tau, T].  Only the NODES
    are used by the models; the weights are re-fitted to the actual kernel.
    """
    best = soe_power_budget(1.0 - alpha, tau, T, n_max)
    return best["s"], best


# ---------------------------------------------------------------------------
# turning nodes into a (fixed-pole) discrete linear system
# ---------------------------------------------------------------------------
def zoh_coeffs(s, tau: float):
    """Zero-order-hold discretisation of h' = -s h + f on step tau.

    h[n] = a h[n-1] + b f[n],   a = exp(-s tau),  b = (1 - a)/s.
    """
    s = np.asarray(s, dtype=np.float64)
    a = np.exp(-s * tau)
    b = -np.expm1(-s * tau) / s
    return a, b


def soe_design(s, tau: float, n_index: np.ndarray):
    """Rows of the impulse response of each fixed mode:  M[i,j] = b_j a_j^{n_i-1}."""
    a, b = zoh_coeffs(s, tau)
    n_index = np.asarray(n_index, dtype=np.float64)
    return b[None, :] * a[None, :] ** (n_index[:, None] - 1.0)


def fit_weights_to_kernel(s, g, tau: float, n_log: int = 600, rcond=None):
    """Least-squares readout c_j so that  sum_j c_j b_j a_j^{n-1} ~ tau*g[n].

    The residuals are scaled by 1/|target| (relative error) and the sample
    points n are log-spaced, because g spans many decades.  This is the
    *training-free* ceiling of the fixed-exponent architecture.
    """
    g = np.asarray(g, dtype=np.float64)
    L = g.size
    n_all = np.arange(1, L + 1)
    idx = np.unique(np.round(np.exp(np.linspace(0.0, np.log(L), n_log))).astype(int))
    idx = idx[(idx >= 1) & (idx <= L)]
    n_s = n_all[idx - 1]
    target = tau * g[idx - 1]
    M = soe_design(s, tau, n_s)
    wgt = 1.0 / np.abs(target)
    c, *_ = np.linalg.lstsq(M * wgt[:, None], target * wgt, rcond=rcond)
    return c


def soe_apply(s, c, f, tau: float):
    """Run the fixed-pole model: y_n = sum_j c_j h_j[n] (ZOH recursion)."""
    s = np.asarray(s, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    squeeze = f.ndim == 1
    f = np.atleast_2d(f)
    a, b = zoh_coeffs(s, tau)
    y = np.zeros_like(f)
    for j in range(s.size):
        y += c[j] * lfilter([b[j]], [1.0, -a[j]], f, axis=1)
    return y[0] if squeeze else y
