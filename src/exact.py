r"""Exact (quadrature-accurate) references for the fractional relaxation operator.

For y(0) = 0 the solution of  D^a y = -lam y + f  is a causal convolution

    y(t) = \int_0^t K(t-r) f(r) dr,     K(t) = t^{a-1} E_{a,a}(-lam t^a),

whose Laplace transform is  \hat K(p) = 1/(p^a + lam).  That transform is a
Stieltjes function, so K is completely monotone and admits an exact
sum-of-exponentials (spectral) representation

    K(t) = \int_0^\infty e^{-s t} mu(s) ds,
    mu(s) = -1/pi * Im \hat K(s e^{i pi})
          = (1/pi) * s^a sin(pi a) / (s^{2a} + 2 lam s^a cos(pi a) + lam^2).

Two facts drive the whole study:

  * mu(s) ~ sin(pi a)/(pi lam^2) * s^a        as s -> 0   =>  K(t) ~ a/(G(1-a) lam^2) t^{-1-a}
  * mu(s) ~ sin(pi a)/pi * s^{-a}             as s -> inf =>  K(t) ~ t^{a-1}/G(a)

i.e. the memory kernel is an *algebraic* (power-law) tail, representable only by
a continuum of relaxation rates s spread over the whole band [1/T, 1/tau].
Discretising the s-integral is exactly what Jiang's formula does (see jiang.py)
and exactly what a diagonal SSM (S4D) does with learned rates.

Everything here is computed with scipy quadrature in log s, which is accurate
to ~1e-12 and is used only to validate the L1 solver and the SOE nodes.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.special import gamma as _gamma


def spectral_density(s, alpha: float, lam: float):
    """mu(s), the spectral (relaxation-rate) density of the kernel K."""
    s = np.asarray(s, dtype=np.float64)
    sa = s ** alpha
    num = np.sin(np.pi * alpha) * sa
    den = sa ** 2 + 2.0 * lam * sa * np.cos(np.pi * alpha) + lam ** 2
    return num / (np.pi * den)


def _log_quad(fn, xa: float, xb: float, limit: int = 400) -> float:
    val, _ = quad(fn, xa, xb, limit=limit, epsabs=1e-14, epsrel=1e-12)
    return val


def kernel(t, alpha: float, lam: float):
    """K(t) = t^{a-1} E_{a,a}(-lam t^a) via the spectral integral."""
    t = np.atleast_1d(np.asarray(t, dtype=np.float64))
    out = np.empty_like(t)
    for i, ti in enumerate(t):
        if ti <= 0:
            out[i] = np.inf
            continue
        # s = e^x; integrand decays like e^{(1+a)x} at -inf and
        # double-exponentially at +inf.
        xb = np.log(40.0 / ti) + 5.0
        fn = lambda x, ti=ti: np.exp(-ti * np.exp(x)) * spectral_density(np.exp(x), alpha, lam) * np.exp(x)
        out[i] = _log_quad(fn, -60.0, xb)
    return out


def step_response(t, alpha: float, lam: float):
    """y(t) for f == 1, i.e. (1 - E_a(-lam t^a))/lam, via the spectral integral.

        y(t) = \int_0^\infty mu(s) (1 - e^{-s t}) / s ds
    """
    t = np.atleast_1d(np.asarray(t, dtype=np.float64))
    out = np.empty_like(t)
    for i, ti in enumerate(t):
        if ti <= 0:
            out[i] = 0.0
            continue
        # in x = log s the integrand is mu(e^x) (1 - e^{-t e^x}), tail ~ e^{-a x}
        fn = lambda x, ti=ti: spectral_density(np.exp(x), alpha, lam) * (
            -np.expm1(-ti * np.exp(x)))
        out[i] = _log_quad(fn, -60.0, 60.0 / alpha + np.log(1.0 / ti))
    return out


def mittag_leffler_relaxation(t, alpha: float, lam: float):
    """E_a(-lam t^a) = 1 - lam * step_response(t)."""
    return 1.0 - lam * step_response(t, alpha, lam)


def kernel_asymptotes(t, alpha: float, lam: float):
    """(small-t, large-t) power laws of K, for plots."""
    t = np.asarray(t, dtype=np.float64)
    small = t ** (alpha - 1.0) / _gamma(alpha)
    large = alpha / (_gamma(1.0 - alpha) * lam ** 2) * t ** (-1.0 - alpha)
    return small, large


def check_spectral_density(alpha: float, lam: float, ps=(0.03, 0.3, 3.0, 30.0)):
    """Verify  \int mu(s)/(p+s) ds == 1/(p^a + lam)  (Laplace of the SOE rep)."""
    errs = []
    for p in ps:
        fn = lambda x: spectral_density(np.exp(x), alpha, lam) * np.exp(x) / (p + np.exp(x))
        got = _log_quad(fn, -60.0, 60.0 / alpha)
        want = 1.0 / (p ** alpha + lam)
        errs.append(abs(got - want) / abs(want))
    return float(max(errs))
