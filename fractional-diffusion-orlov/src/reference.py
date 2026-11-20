"""Analytic continuum reference, independent of the numerical solver."""
import numpy as np
from pymittagleffler import mittag_leffler


def exact_sine_solution(x, t, alpha, coefficients=(1.0,)):
    x = np.asarray(x, dtype=np.float64)
    t = np.atleast_1d(np.asarray(t, dtype=np.float64))
    result = np.zeros((len(t), len(x)), dtype=np.float64)
    for k, amplitude in enumerate(coefficients, start=1):
        z = -(k*np.pi)**2 * t**alpha
        values = np.asarray(mittag_leffler(z, alpha, 1.0))
        if np.max(np.abs(values.imag)) > 1e-11:
            raise RuntimeError('Mittag-Leffler reference has a non-negligible imaginary part')
        values = values.real
        values[t == 0] = 1.0
        result += amplitude * values[:, None] * np.sin(k*np.pi*x)[None, :]
    result[:, [0, -1]] = 0
    if not np.all(np.isfinite(result)):
        raise RuntimeError('non-finite Mittag-Leffler reference')
    return result
