import numpy as np


def _first_mode(u0):
    initial = np.asarray(u0, dtype=float)
    if initial.ndim != 2 or initial.shape[1] < 3 or not np.isfinite(initial).all():
        raise ValueError('u0 must be a finite [batch,x] array')
    basis = np.sin(np.pi*np.linspace(0, 1, initial.shape[1]))
    basis[[0, -1]] = 0
    amplitude = initial @ basis / (basis @ basis)
    return amplitude, basis


def fit_first_mode(u0, alpha, targets, degree=5):
    amplitude, basis = _first_mode(u0)
    a = np.asarray(alpha, dtype=float)
    target = np.asarray(targets, dtype=float)
    if (a.shape != amplitude.shape or target.ndim != 3
            or target.shape[0] != len(a) or target.shape[2] != len(basis)
            or not np.isfinite(a).all() or not np.isfinite(target).all()
            or not isinstance(degree, int) or degree < 0):
        raise ValueError('invalid baseline training data')
    design = amplitude[:, None]*np.polynomial.chebyshev.chebvander((a-.55)/.35, degree)
    projection = target @ basis / (basis @ basis)
    weights = 1/np.maximum(np.linalg.norm(target, axis=-1), 1e-12)
    return np.stack([np.linalg.lstsq(design*weights[:, j, None],
                                    projection[:, j]*weights[:, j], rcond=None)[0]
                     for j in range(target.shape[1])])


def predict_first_mode(u0, alpha, coefficients):
    amplitude, basis = _first_mode(u0)
    a = np.asarray(alpha, dtype=float)
    coef = np.asarray(coefficients, dtype=float)
    if a.shape != amplitude.shape or coef.ndim != 2 or not np.isfinite(coef).all():
        raise ValueError('invalid baseline prediction data')
    design = np.polynomial.chebyshev.chebvander((a-.55)/.35, coef.shape[1]-1)
    damping = design @ coef.T
    return amplitude[:, None, None]*damping[:, :, None]*basis
