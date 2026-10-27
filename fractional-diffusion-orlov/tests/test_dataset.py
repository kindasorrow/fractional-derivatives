import tempfile
import unittest
from pathlib import Path
import numpy as np
from src.generate_dataset import audit, fit_output_normalization


class DatasetTests(unittest.TestCase):
    def test_normalization_uses_train_only(self):
        u = np.array([[[1., 3.], [2., 6.]], [[100., 100.], [200., 200.]]])
        mean, scale = fit_output_normalization(u, np.array(['train', 'test_random']))
        np.testing.assert_allclose(mean, [[2.], [4.]])
        np.testing.assert_allclose(scale, [[1.], [2.]])
        normalized = (u - mean) / scale
        np.testing.assert_allclose(normalized[0], [[-1., 1.], [-1., 1.]])
        np.testing.assert_allclose(normalized * scale + mean, u)

    def test_audit_rejects_invalid_shapes_with_value_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'invalid.npz'
            np.savez(path, u0=np.zeros((1, 3)), alpha=np.array([.5]),
                     uT=np.zeros((1, 3)), split=np.array(['train']))
            with self.assertRaisesRegex(ValueError, 'shape'):
                audit(path)
