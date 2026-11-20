"""Independent assembly, known special function, and modal equivalence checks."""
import unittest
import numpy as np
from scipy.special import gamma, erfcx
from src.solver import solve_l1, solve_l1_sine_batch, sine_basis, l1_weights
from src.reference import exact_sine_solution


class SolverTests(unittest.TestCase):
    def test_independent_dense_assembly(self):
        # An independently expanded history formula on a small spatial grid.
        nodes, steps, a = 9, 13, 0.37
        u0 = np.array([0, 1, -.3, .7, -.1, .8, -.4, .2, 0.])
        actual = solve_l1(u0, a, n_steps=steps).u
        r = gamma(2-a)*(1/steps)**a*(nodes-1)**2
        matrix = np.diag(np.full(nodes-2, 1+2*r))
        matrix += np.diag(np.full(nodes-3, -r), 1)+np.diag(np.full(nodes-3, -r), -1)
        b = np.diff(np.arange(steps+1, dtype=float)**(1-a))
        expected = np.zeros_like(actual)
        expected[0] = u0
        for n in range(1, steps+1):
            rhs = b[n-1]*u0[1:-1]
            for k in range(1, n):
                rhs += (b[k-1]-b[k])*expected[n-k, 1:-1]
            expected[n, 1:-1] = np.linalg.solve(matrix, rhs)
        np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=1e-12)

    def test_sine_batch_matches_general_solver(self):
        rng = np.random.default_rng(7)
        coef = rng.normal(size=(5, 8))
        alpha = np.array([.2, .35, .5, .75, .9])
        initial = coef @ sine_basis()
        fast = solve_l1_sine_batch(coef, alpha)
        for i, a in enumerate(alpha):
            general = solve_l1(initial[i], a).u[-1]
            np.testing.assert_allclose(fast[i], general, rtol=1e-10, atol=2e-12)

    def test_mittag_leffler_half_order_identity(self):
        t = np.linspace(0, 1, 51)
        actual = exact_sine_solution([0, .5, 1], t, .5)[:, 1]
        np.testing.assert_allclose(actual, erfcx(np.pi**2*np.sqrt(t)), rtol=2e-12, atol=1e-13)

    def test_zero_initial_condition(self):
        sol = solve_l1(np.zeros(128), .2)
        np.testing.assert_array_equal(sol.u, 0)

    def test_linearity_and_boundaries(self):
        x = np.linspace(0, 1, 33)
        u, v = np.sin(np.pi*x), np.sin(3*np.pi*x)
        left = solve_l1(2*u-.3*v, .75, 40).u
        right = 2*solve_l1(u, .75, 40).u-.3*solve_l1(v, .75, 40).u
        np.testing.assert_allclose(left, right, atol=2e-14)
        np.testing.assert_array_equal(left[:, [0,-1]], 0)

    def test_exact_initial_condition(self):
        x = np.linspace(0, 1, 128)
        u0 = sine_basis()[0]
        np.testing.assert_array_equal(solve_l1(u0, .35, 5).u[0], u0)

    def test_weights_positive_decreasing(self):
        b = l1_weights(np.array([.2,.5,.9]), 500)
        self.assertTrue(np.all(b > 0))
        self.assertTrue(np.all(np.diff(b, axis=1) < 0))

    def test_invalid_input(self):
        for a in [0, 1, -1, np.nan]:
            with self.assertRaises(ValueError):
                solve_l1(np.zeros(8), a)
        with self.assertRaises(ValueError):
            solve_l1(np.ones(8), .5)
        with self.assertRaises(ValueError):
            solve_l1(np.zeros(8), .5, n_steps=0)


if __name__ == '__main__':
    unittest.main()
