"""Цветной шум (1/f^beta) из белого шума через дробный интеграл."""

from .generators import (
    gl_weights,
    design_exp_bank,
    pink_fft,
    pink_gl,
    pink_exp,
    pink_kellett,
)
from .analysis import spectrum, fit_slope, timeit

__all__ = [
    "gl_weights", "design_exp_bank", "pink_fft", "pink_gl", "pink_exp",
    "pink_kellett", "spectrum", "fit_slope", "timeit",
]