import sys
from pathlib import Path

# Добавляем папку src в sys.path, чтобы тесты находили пакет pinknoise
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest

from pinknoise import (design_exp_bank, fit_slope, gl_weights, pink_exp,
                       pink_fft, pink_gl, spectrum)

N = 2 ** 16


@pytest.fixture(scope="module")
def white():
    return np.random.default_rng(0).standard_normal(N)


@pytest.mark.parametrize("beta", [0.5, 1.0, 1.5])
def test_fft_slope(white, beta):
    assert fit_slope(*spectrum(pink_fft(white, beta))) == pytest.approx(-beta, abs=0.08)


@pytest.mark.parametrize("beta", [0.5, 1.0, 1.5])
def test_gl_slope(white, beta):
    assert fit_slope(*spectrum(pink_gl(white, beta))) == pytest.approx(-beta, abs=0.08)


@pytest.mark.parametrize("beta", [0.5, 1.0, 1.5])
def test_exp_bank_slope(white, beta):
    a, b = design_exp_bank(beta)
    assert fit_slope(*spectrum(pink_exp(white, a, b))) == pytest.approx(-beta, abs=0.08)


def test_gl_weights_gamma_one_is_running_sum():
    # gamma = 1 -> обычный интеграл -> все веса равны 1
    assert np.allclose(gl_weights(1.0, 10), 1.0)


def test_gl_weights_gamma_half():
    w = gl_weights(0.5, 4)
    assert np.allclose(w, [1, 0.5, 0.375, 0.3125])