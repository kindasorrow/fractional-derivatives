import unittest
import numpy as np
from scipy.special import gamma
from pymittagleffler import mittag_leffler
from src.advanced_solver import graded_time_grid, build_soe_kernel, solve_graded_l1, solve_soe
from src.solver import solve_l1


class AdvancedSolverTests(unittest.TestCase):
    def test_nonuniform_l1_matches_independent_dense_assembly(self):
        initial = np.array([0., .3, -.7, .2, 0.])
        alpha = .35
        actual = solve_graded_l1(initial, alpha, n_steps=15)
        dt = np.diff(actual.t)
        expected = np.zeros_like(actual.u)
        expected[0] = initial
        laplacian = (np.diag(np.full(3, -2.)) + np.diag(np.ones(2), 1)
                     + np.diag(np.ones(2), -1)) * 16
        for n in range(1, 16):
            weights = ((actual.t[n] - actual.t[:n])**(1-alpha)
                       - (actual.t[n] - actual.t[1:n+1])**(1-alpha)) / (gamma(2-alpha)*dt[:n])
            history = weights[:-1] @ np.diff(expected[:n, 1:-1], axis=0)
            expected[n, 1:-1] = np.linalg.solve(weights[-1]*np.eye(3)-laplacian,
                                               weights[-1]*expected[n-1, 1:-1]-history)
        np.testing.assert_allclose(actual.u, expected, rtol=2e-11, atol=2e-13)

    def test_uniform_mesh_reproduces_original_l1(self):
        initial = np.array([0., .3, -.7, .2, 0.])
        for alpha in (.2, .5, .9):
            with self.subTest(alpha=alpha):
                actual = solve_graded_l1(initial, alpha, n_steps=80, grading=1)
                expected = solve_l1(initial, alpha, n_steps=80)
                np.testing.assert_allclose(actual.u, expected.u, rtol=2e-11, atol=2e-13)

    def test_positive_kernel_on_extreme_graded_intervals(self):
        for alpha in (.2, .35, .5, .75, .9):
            with self.subTest(alpha=alpha):
                times = graded_time_grid(alpha, 500)
                kernel = build_soe_kernel(alpha, np.diff(times).min(), 1., tolerance=1e-8)
                points = np.geomspace(np.diff(times).min(), 1., 1103)
                actual = np.exp(-points[:, None]*kernel.rates) @ kernel.weights
                expected = points**(-alpha)/gamma(1-alpha)
                self.assertLess(np.max(np.abs(actual/expected-1)), 1e-8)
                self.assertTrue(np.all(kernel.rates > 0))
                self.assertTrue(np.all(kernel.weights > 0))

    def test_soe_matches_full_l1_on_uniform_and_graded_meshes(self):
        initial = np.array([0., .3, -.7, .2, .1, 0.])
        for alpha in (.2, .35, .5, .75, .9):
            for grading in (1., None):
                with self.subTest(alpha=alpha, grading=grading):
                    expected = solve_graded_l1(initial, alpha, n_steps=120, grading=grading)
                    actual = solve_soe(initial, alpha, n_steps=120, grading=grading,
                                       tolerance=1e-9)
                    np.testing.assert_array_equal(actual.t, expected.t)
                    np.testing.assert_allclose(actual.u, expected.u, rtol=2e-7, atol=3e-10)
                    np.testing.assert_array_equal(actual.u[:, [0, -1]], 0)

    def test_invalid_inputs(self):
        for solver in (solve_graded_l1, solve_soe):
            for kwargs in ({'alpha':0}, {'alpha':1}, {'alpha':np.nan},
                           {'alpha':.5, 'n_steps':0}, {'alpha':.5, 'grading':0},
                           {'alpha':.5, 'T':0}):
                with self.subTest(solver=solver.__name__, kwargs=kwargs), self.assertRaises(ValueError):
                    solver(np.zeros(5), **kwargs)
            with self.assertRaises(ValueError):
                solver(np.ones(5), .5)
        with self.assertRaises(ValueError):
            build_soe_kernel(.5, .1, 1., tolerance=0)

    def test_grading_reduces_full_trajectory_temporal_error(self):
        x = np.linspace(0, 1, 9)
        initial = np.sin(np.pi*x)
        eigenvalue = 4*8**2*np.sin(np.pi/16)**2
        for alpha in (.2, .35, .5, .75, .9):
            with self.subTest(alpha=alpha):
                errors = []
                for grading in (1., None):
                    solution = solve_graded_l1(initial, alpha, n_steps=256, grading=grading)
                    amplitude = np.real(mittag_leffler(-eigenvalue*solution.t**alpha, alpha, 1))
                    errors.append(np.max(np.abs(solution.u-amplitude[:, None]*initial)))
                self.assertLess(errors[1], .65*errors[0])


if __name__ == '__main__':
    unittest.main()
