import unittest
import numpy as np
from src.solver import sine_basis, solve_l1, solve_l1_batch, solve_l1_sine_batch


class BatchTests(unittest.TestCase):
    def test_arbitrary_initial_conditions_match_general_trajectory(self):
        rng = np.random.default_rng(34)
        for n_x in (3, 17, 34):
            initial = rng.normal(size=(3, n_x))
            initial[:, [0, -1]] = 0
            alpha = np.array([.2, .5, .9])
            steps = np.array([0, 1, 3, 17])
            actual = solve_l1_batch(initial, alpha, n_steps=17, T=.1,
                                    sample_steps=steps)
            self.assertEqual(actual.shape, (3, 4, n_x))
            for i, a in enumerate(alpha):
                expected = solve_l1(initial[i], a, n_steps=17, T=.1).u[steps]
                np.testing.assert_allclose(actual[i], expected, rtol=1e-10, atol=2e-12)
            np.testing.assert_array_equal(actual[:, :, [0, -1]], 0)

    def test_final_state_compatibility(self):
        coef = np.array([[.3, -.7, .4], [.5, .2, -.1]])
        alpha = np.array([.35, .75])
        initial = coef @ sine_basis(25, 3)
        final = solve_l1_sine_batch(coef, alpha, n_x=25, n_steps=20)
        sampled = solve_l1_sine_batch(coef, alpha, n_x=25, n_steps=20,
                                     sample_steps=[0, 5, 20])
        self.assertEqual(final.shape, (2, 25))
        np.testing.assert_array_equal(final, sampled[:, -1])
        np.testing.assert_allclose(sampled[:, 0], initial, atol=1e-14)
        np.testing.assert_allclose(solve_l1_batch(initial, alpha, n_steps=20),
                                   final, rtol=1e-10, atol=2e-12)

    def test_invalid_sample_steps(self):
        for steps in ([], [-1], [21], [1, 1], [4, 2], [1.0], [True], [[1]]):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                solve_l1_batch(np.zeros((1, 9)), [.5], n_steps=20,
                               sample_steps=steps)

    def test_invalid_initial_conditions_and_alpha(self):
        for initial, alpha in ((np.zeros(9), [.5]), (np.zeros((1, 2)), [.5]),
                               (np.ones((1, 9)), [.5]),
                               (np.full((1, 9), np.nan), [.5]),
                               (np.zeros((1, 9)), [.5, .6]),
                               (np.zeros((1, 9)), [np.nan]),
                               (np.zeros((1, 9)), [1.0])):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                solve_l1_batch(initial, alpha, n_steps=2)


if __name__ == '__main__':
    unittest.main()
