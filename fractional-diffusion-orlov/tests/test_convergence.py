import unittest
import numpy as np
from src import validate


class ConvergenceTests(unittest.TestCase):
    def test_temporal_convergence_on_nondefault_grid(self):
        time_steps = (125, 250, 500, 1000)
        for alpha in (.2, .35, .5, .75, .9):
            with self.subTest(alpha=alpha):
                temporal, _ = validate.convergence_errors(
                    alpha, n_x=65, time_steps=time_steps, space_nodes=())
                self.assertTrue(np.all(np.diff(temporal) < 0))
                time_order = np.polyfit(np.log(1 / np.array(time_steps)), np.log(temporal), 1)[0]
                self.assertGreater(time_order, .9)
                self.assertLess(time_order, 1.15)

    def test_spatial_convergence(self):
        space_nodes = (32, 64, 128, 256)
        for alpha in (.2, .35, .5, .75, .9):
            with self.subTest(alpha=alpha):
                _, spatial = validate.convergence_errors(
                    alpha, time_steps=(), space_nodes=space_nodes)
                self.assertTrue(np.all(np.diff(spatial) < 0))
                space_order = np.polyfit(np.log(1 / (np.array(space_nodes) - 1)), np.log(spatial), 1)[0]
                self.assertGreater(space_order, 1.95)
                self.assertLess(space_order, 2.05)


if __name__ == '__main__':
    unittest.main()
