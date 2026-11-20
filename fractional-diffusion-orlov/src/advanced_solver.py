from dataclasses import dataclass
import numpy as np
from scipy.fft import dst, idst
from scipy.special import gamma, gammainccinv
from .solver import Solution, _check_grid, l1_weights


@dataclass
class ExponentialKernel:
    rates: np.ndarray
    weights: np.ndarray
    alpha: float
    minimum_lag: float
    maximum_lag: float
    tolerance: float
    measured_relative_error: float


def graded_time_grid(alpha, n_steps=500, T=1., grading=None):
    if np.ndim(alpha) != 0:
        raise ValueError('alpha must be scalar')
    l1_weights(alpha, 1)
    _check_grid(3, n_steps, T)
    grading = (2-alpha)/alpha if grading is None else grading
    if not np.isfinite(grading) or grading < 1:
        raise ValueError('grading must be finite and at least one')
    times = T*(np.arange(n_steps+1, dtype=float)/n_steps)**grading
    if np.any(np.diff(times) <= 0):
        raise ValueError('time mesh is not representable in float64')
    return times


def build_soe_kernel(alpha, minimum_lag, maximum_lag, tolerance=1e-8):
    if np.ndim(alpha) != 0:
        raise ValueError('alpha must be scalar')
    l1_weights(alpha, 1)
    if (not np.isfinite(minimum_lag) or not np.isfinite(maximum_lag)
            or not 0 < minimum_lag <= maximum_lag):
        raise ValueError('kernel interval must satisfy 0 < minimum_lag <= maximum_lag')
    if not np.isfinite(tolerance) or not 1e-13 <= tolerance < .1:
        raise ValueError('tolerance must lie in [1e-13, .1)')
    lower = (np.log(tolerance*gamma(1+alpha)/32)/alpha
             - np.log(maximum_lag))
    upper = np.log(gammainccinv(alpha, tolerance/32))-np.log(minimum_lag)
    if lower < -700 or upper > 700:
        raise ValueError('kernel interval and alpha exceed float64 quadrature range')
    points = np.geomspace(minimum_lag, maximum_lag, 769)
    exact = points**(-alpha)/gamma(1-alpha)
    for refinement in range(7):
        count = int(np.ceil((upper-lower)*2**refinement))+1
        nodes = np.linspace(lower, upper, count)
        rates = np.exp(nodes)
        weights = np.sin(np.pi*alpha)/np.pi * (nodes[1]-nodes[0])*np.exp(alpha*nodes)
        weights[[0, -1]] *= .5
        actual = np.exp(-points[:, None]*rates) @ weights
        error = float(np.max(np.abs(actual/exact-1)))
        if error <= tolerance/2:
            return ExponentialKernel(rates, weights, float(alpha), float(minimum_lag),
                                     float(maximum_lag), float(tolerance), error)
    raise RuntimeError('SOE quadrature did not meet the requested sampled tolerance')


def _prepare(u0, alpha, n_steps, T, grading):
    initial = np.asarray(u0, dtype=float)
    if initial.ndim != 1:
        raise ValueError('u0 must be one-dimensional')
    _check_grid(len(initial), n_steps, T)
    if not np.isfinite(initial).all():
        raise ValueError('u0 must be finite')
    if not np.allclose(initial[[0, -1]], 0, rtol=0, atol=1e-12):
        raise ValueError('u0 must have zero boundaries')
    times = graded_time_grid(alpha, n_steps, T, grading)
    modes = np.arange(1, len(initial)-1)
    eigenvalues = 4*(len(initial)-1)**2*np.sin(modes*np.pi/(2*(len(initial)-1)))**2
    coefficients = dst(initial[1:-1], type=1, norm='ortho')
    return initial, times, coefficients, eigenvalues


def _solution(initial, times, values):
    u = np.zeros((len(times), len(initial)))
    u[:, 1:-1] = idst(values, type=1, norm='ortho', axis=-1)
    u[0, 1:-1] = initial[1:-1]
    return Solution(np.linspace(0, 1, len(initial)), times, u)


def solve_graded_l1(u0, alpha, n_steps=500, T=1., grading=None):
    initial, times, coefficients, eigenvalues = _prepare(u0, alpha, n_steps, T, grading)
    dt = np.diff(times)
    values = np.empty((n_steps+1, len(coefficients)))
    values[0] = coefficients
    delta = np.empty((n_steps, len(coefficients)))
    for n in range(1, n_steps+1):
        local = dt[n-1]**(-alpha)/gamma(2-alpha)
        history = np.zeros_like(coefficients)
        if n > 1:
            lag = times[n]-times[1:n]
            weights = lag**(1-alpha)*np.expm1((1-alpha)*np.log1p(dt[:n-1]/lag))
            weights /= gamma(2-alpha)*dt[:n-1]
            history = np.einsum('j,jk->k', weights, delta[:n-1], optimize=False)
        values[n] = (values[n-1]-history/local)/(1+eigenvalues/local)
        delta[n-1] = values[n]-values[n-1]
    return _solution(initial, times, values)


def solve_soe(u0, alpha, n_steps=500, T=1., grading=None, tolerance=1e-8,
              kernel=None):
    initial, times, coefficients, eigenvalues = _prepare(u0, alpha, n_steps, T, grading)
    dt = np.diff(times)
    if kernel is None:
        kernel = build_soe_kernel(alpha, dt.min(), T, tolerance)
    if (kernel.alpha != alpha or kernel.minimum_lag > dt.min()
            or kernel.maximum_lag < T or kernel.tolerance > tolerance):
        raise ValueError('kernel does not cover the mesh, alpha or tolerance')
    values = np.empty((n_steps+1, len(coefficients)))
    values[0] = coefficients
    history_modes = np.zeros((len(kernel.rates), len(coefficients)))
    previous_delta = np.zeros_like(coefficients)
    for n in range(1, n_steps+1):
        if n > 1:
            decay = np.exp(-kernel.rates*dt[n-1])
            z = kernel.rates*dt[n-2]
            integral = -np.expm1(-z)/z
            history_modes = decay[:, None]*(history_modes+integral[:, None]*previous_delta)
        history = np.einsum('j,jk->k', kernel.weights, history_modes, optimize=False)
        local = dt[n-1]**(-alpha)/gamma(2-alpha)
        values[n] = (values[n-1]-history/local)/(1+eigenvalues/local)
        previous_delta = values[n]-values[n-1]
    return _solution(initial, times, values)
