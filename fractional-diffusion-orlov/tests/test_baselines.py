import unittest
import numpy as np
from src.baselines import fit_first_mode, predict_first_mode


class BaselineTests(unittest.TestCase):
    def test_recovers_polynomial_damping_without_test_targets(self):
        x = np.linspace(0, 1, 65)
        basis = np.sin(np.pi*x)
        basis[[0, -1]] = 0
        alpha = np.linspace(.2, .9, 20)
        amplitudes = np.linspace(-1, 1, 20)
        initial = amplitudes[:, None]*basis
        damping = np.stack([1-alpha, (1-alpha)**2], axis=1)
        target = initial[:, None, :]*damping[:, :, None]
        coefficients = fit_first_mode(initial, alpha, target, degree=2)
        new_alpha = np.array([.35, .75])
        new_initial = np.stack([basis, -basis])
        expected = new_initial[:, None, :]*np.stack([1-new_alpha, (1-new_alpha)**2], axis=1)[:, :, None]
        actual = predict_first_mode(new_initial, new_alpha, coefficients)
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_higher_modes_do_not_change_prediction(self):
        x = np.linspace(0, 1, 65)
        initial = np.stack([np.sin(np.pi*x), np.sin(np.pi*x)+2*np.sin(4*np.pi*x)])
        initial[:, [0, -1]] = 0
        actual = predict_first_mode(initial, np.array([.5, .5]), np.array([[.1]]))
        np.testing.assert_allclose(actual[0], actual[1], atol=1e-14)
