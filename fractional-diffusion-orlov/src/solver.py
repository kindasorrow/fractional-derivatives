"""Solve D_t^alpha u = u_xx with homogeneous Dirichlet boundaries.

128 grid points means 126 interior unknowns and two boundary points.
All public solution arrays include the initial time and both boundaries.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg.lapack import dpttrf, dpttrs
from scipy.special import gamma


@dataclass
class Solution:
    x: np.ndarray
    t: np.ndarray
    u: np.ndarray


def l1_weights(alpha, n_steps):
    """b_j = (j+1)^(1-alpha) - j^(1-alpha), computed stably."""
    a = np.asarray(alpha, dtype=np.float64)
    if np.any(~np.isfinite(a)) or np.any((a <= 0) | (a >= 1)):
        raise ValueError('alpha must be finite and strictly between 0 and 1')
    if not isinstance(n_steps, (int, np.integer)) or n_steps < 1:
        raise ValueError('n_steps must be a positive integer')
    j = np.arange(1, n_steps, dtype=np.float64)
    p = 1 - a[..., None]
    tail = j**p * np.expm1(p * np.log1p(1 / j))
    return np.concatenate((np.ones(a.shape + (1,)), tail), axis=-1)


def _check_grid(n_x, n_steps, T):
    if not isinstance(n_x, (int, np.integer)) or n_x < 3:
        raise ValueError('n_x must be an integer >= 3')
    if not isinstance(n_steps, (int, np.integer)) or n_steps < 1:
        raise ValueError('n_steps must be a positive integer')
    if not np.isfinite(T) or T <= 0:
        raise ValueError('T must be positive and finite')


def solve_l1(u0, alpha, n_steps=500, T=1.0):
    """General spatial solver; factor the SPD tridiagonal matrix once.

    (I - c D_xx) u^n = u^(n-1) - sum b_j delta_u^(n-j),
    c = Gamma(2-alpha) dt^alpha, j=1,...,n-1.
    """
    u0 = np.asarray(u0, dtype=np.float64)
    if u0.ndim != 1:
        raise ValueError('u0 must be a one-dimensional array')
    n_x = len(u0)
    _check_grid(n_x, n_steps, T)
    if not np.all(np.isfinite(u0)):
        raise ValueError('u0 must be finite')
    if not np.allclose(u0[[0, -1]], 0, rtol=0, atol=1e-12):
        raise ValueError('u0 must satisfy zero Dirichlet boundaries')
    if np.ndim(alpha) != 0:
        raise ValueError('solve_l1 expects one scalar alpha')
    b = l1_weights(alpha, n_steps)
    dx = 1 / (n_x - 1)
    c = gamma(2 - alpha) * (T / n_steps)**alpha
    r = c / dx**2
    diag, off, info = dpttrf(np.full(n_x - 2, 1 + 2*r),
                            np.full(n_x - 3, -r))
    if info != 0:
        raise RuntimeError(f'tridiagonal factorization failed: {info}')
    u = np.zeros((n_steps + 1, n_x), dtype=np.float64)
    u[0, 1:-1] = u0[1:-1]
    delta = np.empty((n_steps, n_x - 2), dtype=np.float64)
    for n in range(1, n_steps + 1):
        rhs = u[n-1, 1:-1].copy()
        if n > 1:
            rhs -= np.einsum('j,jk->k', b[n-1:0:-1], delta[:n-1], optimize=False)
        value, info = dpttrs(diag, off, rhs[:, None])
        if info != 0:
            raise RuntimeError(f'tridiagonal solve failed: {info}')
        u[n, 1:-1] = value[:, 0]
        delta[n-1] = u[n, 1:-1] - u[n-1, 1:-1]
    return Solution(np.linspace(0, 1, n_x), np.linspace(0, T, n_steps+1), u)


def sine_basis(n_x=128, n_modes=8):
    if not 1 <= n_modes <= n_x - 2:
        raise ValueError('n_modes must lie between 1 and n_x-2')
    x = np.linspace(0, 1, n_x)
    basis = np.sin(np.pi * np.arange(1, n_modes+1)[:, None] * x)
    basis[:, [0, -1]] = 0
    return basis


def solve_l1_sine_batch(coefficients, alpha, n_x=128, n_steps=500, T=1.0):
    """Return final states for finite sine sums, using the SAME FD/L1 scheme.

    Uses discrete eigenvalues 4/dx^2 sin^2(k*pi/(2*(n_x-1))),
    not continuum eigenvalues (k*pi)^2 or analytic Mittag-Leffler labels.
    O(batch * modes * n_steps^2), with no temporal approximation added.
    """
    _check_grid(n_x, n_steps, T)
    coef = np.asarray(coefficients, dtype=np.float64)
    a = np.asarray(alpha, dtype=np.float64)
    if coef.ndim != 2 or a.shape != (coef.shape[0],):
        raise ValueError('coefficients must be [batch,modes], alpha [batch]')
    if not np.all(np.isfinite(coef)):
        raise ValueError('coefficients must be finite')
    basis = sine_basis(n_x, coef.shape[1])
    b = l1_weights(a, n_steps)
    k = np.arange(1, coef.shape[1]+1)
    lam = 4 * (n_x-1)**2 * np.sin(k*np.pi/(2*(n_x-1)))**2
    c = gamma(2-a) * (T/n_steps)**a
    denominator = 1 + c[:, None] * lam
    previous = coef.copy()
    delta = np.empty((n_steps,) + coef.shape, dtype=np.float64)
    for n in range(1, n_steps+1):
        rhs = previous.copy()
        if n > 1:
            rhs -= np.einsum('bj,jbk->bk', b[:, n-1:0:-1], delta[:n-1], optimize=False)
        current = rhs / denominator
        delta[n-1] = current - previous
        previous = current
    final = previous @ basis
    final[:, [0, -1]] = 0
    return final
