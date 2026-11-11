r"""L1 scheme for the Caputo fractional relaxation equation.

    D^a y(t) = -lam * y(t) + f(t),        y(0) = y0,        0 < a < 1

with the Caputo derivative

    D^a y(t) = 1/G(1-a) \int_0^t (t-s)^{-a} y'(s) ds.

On a uniform grid t_n = n*tau the L1 (Oldham-Spanier) approximation replaces
y' by a backward difference on every subinterval:

    D^a y(t_n) ~ tau^{-a}/G(2-a) * sum_{k=0}^{n-1} b_k (y_{n-k} - y_{n-k-1}),
    b_k = (k+1)^{1-a} - k^{1-a}.

Substituting into the equation and solving for y_n gives the explicit recursion
used below (all coefficients positive, see `l1_history_weights`):

    (1 + mu*lam) y_n = sum_{m=1}^{n-1} d_m y_{n-m} + b_{n-1} y_0 + mu f_n,
    mu = tau^a G(2-a),   d_m = b_{m-1} - b_m > 0,
    sum_{m=1}^{n-1} d_m + b_{n-1} = 1.

For y0 = 0 the scheme is an exactly causal, linear, time-invariant map f -> y,
so it equals a discrete convolution with its own impulse response g (see
`discrete_kernel`).  `l1_solve_fft` exploits that for O(L log L) data
generation; `l1_solve` is the plain O(L^2) recursion kept as the reference.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gamma as _gamma


def l1_weights(alpha: float, n: int) -> np.ndarray:
    """b_k = (k+1)^(1-alpha) - k^(1-alpha) for k = 0 .. n-1."""
    k = np.arange(n + 1, dtype=np.float64)
    return np.diff(k ** (1.0 - alpha))


def l1_history_weights(alpha: float, n: int):
    """Return (b, d) with d_m = b_{m-1} - b_m (d[0] is unused / NaN)."""
    b = l1_weights(alpha, n)
    d = np.empty(n, dtype=np.float64)
    d[0] = np.nan
    d[1:] = b[:-1] - b[1:]
    return b, d


def l1_solve(f, alpha: float, lam: float, tau: float, y0: float = 0.0) -> np.ndarray:
    """Direct O(L^2) L1 recursion.

    Parameters
    ----------
    f : array (L,) or (B, L)
        Forcing sampled at t_1 .. t_L (t_j = j*tau).  t_0 is not needed.
    Returns
    -------
    y : array of the same shape, the solution at t_1 .. t_L.
    """
    f = np.asarray(f, dtype=np.float64)
    squeeze = f.ndim == 1
    f = np.atleast_2d(f)
    B, L = f.shape

    b, d = l1_history_weights(alpha, L)
    mu = tau ** alpha * _gamma(2.0 - alpha)
    denom = 1.0 + mu * lam

    # drev[i] = d_{L-1-i}; the slice drev[L-n:L-1] is [d_{n-1}, ..., d_1],
    # contiguous, so the history sum is a single BLAS gemv.
    drev = np.ascontiguousarray(d[1:][::-1])

    Y = np.zeros((B, L + 1), dtype=np.float64)
    Y[:, 0] = y0
    for n in range(1, L + 1):
        acc = mu * f[:, n - 1] + b[n - 1] * y0
        if n > 1:
            acc = acc + Y[:, 1:n] @ drev[L - n:L - 1]
        Y[:, n] = acc / denom
    out = Y[:, 1:]
    return out[0] if squeeze else out


def discrete_kernel(alpha: float, lam: float, tau: float, L: int) -> np.ndarray:
    """Impulse response g[1..L] of the L1 scheme, normalised so that

        y = tau * (g conv f),   g[n] ~ K(n*tau)

    where K is the continuous solution kernel (see exact.kernel).  Obtained by
    running the recursion on f = e_1 and dividing by tau (a unit sample of the
    piecewise-constant forcing carries area tau).
    """
    f = np.zeros(L, dtype=np.float64)
    f[0] = 1.0
    return l1_solve(f, alpha, lam, tau, 0.0) / tau


def l1_solve_fft(f, alpha: float, lam: float, tau: float, kernel=None) -> np.ndarray:
    """O(L log L) solve for y0 = 0 via the LTI property (used for data gen)."""
    from scipy.signal import fftconvolve

    f = np.asarray(f, dtype=np.float64)
    squeeze = f.ndim == 1
    f = np.atleast_2d(f)
    L = f.shape[1]
    g = discrete_kernel(alpha, lam, tau, L) if kernel is None else kernel[:L]
    y = tau * fftconvolve(f, g[None, :], mode="full", axes=1)[:, :L]
    return y[0] if squeeze else y
