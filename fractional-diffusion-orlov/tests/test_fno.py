import unittest
import numpy as np
import torch
from src.fno import FNO1d
from src.train_fno import decode_output, fit_normalization, split_training_ids


class FNOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_invalid_physical_inputs_are_rejected(self):
        model = FNO1d(width=4, modes=4, depth=1)
        initial = torch.zeros(1, 9)
        for value in (0., -.1, float('nan'), float('inf')):
            with self.subTest(time=value), self.assertRaises(ValueError):
                model(initial, torch.tensor([.5]), torch.tensor([value]))
        for value in (0., 1., -.1, float('nan'), float('inf')):
            with self.subTest(alpha=value), self.assertRaises(ValueError):
                model(initial, torch.tensor([value]), torch.tensor([.1]))
        with self.assertRaises(ValueError):
            model(torch.full((1, 9), float('nan')), torch.tensor([.5]), torch.tensor([.1]))

    def test_shape_boundaries_and_conditioning(self):
        torch.manual_seed(12)
        model = FNO1d(width=8, modes=6, depth=2)
        initial = torch.sin(torch.linspace(0, torch.pi, 33)).repeat(3, 1)
        alpha = torch.tensor([.2, .8, .2])
        times = torch.tensor([.1, .1, 1.])
        predicted = model(initial, alpha, times)
        self.assertEqual(predicted.shape, initial.shape)
        self.assertTrue(torch.isfinite(predicted).all())
        torch.testing.assert_close(predicted[:, [0, -1]], torch.zeros(3, 2))
        self.assertGreater(torch.max(torch.abs(predicted[0] - predicted[1])).item(), 1e-6)
        self.assertGreater(torch.max(torch.abs(predicted[0] - predicted[2])).item(), 1e-6)
        physical = decode_output(predicted, torch.ones(3, 1), torch.full((3, 1), 2.))
        torch.testing.assert_close(physical[:, [0, -1]], torch.zeros(3, 2))

    def test_small_batch_overfit_and_spectral_gradients(self):
        torch.manual_seed(8)
        model = FNO1d(width=8, modes=6, depth=2)
        x = torch.linspace(0, 1, 17)
        initial = torch.stack([torch.sin(torch.pi*x), torch.sin(2*torch.pi*x)])
        alpha, times = torch.tensor([.3, .7]), torch.tensor([.01, .1])
        target = initial * torch.tensor([[.4], [.15]])
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        first_loss = None
        for _ in range(100):
            optimizer.zero_grad()
            loss = ((model(initial, alpha, times) - target)**2).mean()
            if first_loss is None:
                first_loss = loss.item()
            loss.backward()
            self.assertIsNotNone(model.blocks[0].spectral.weight.grad)
            self.assertTrue(torch.isfinite(model.blocks[0].spectral.weight.grad).all())
            optimizer.step()
        self.assertLess(loss.item(), first_loss * .1)

    def test_validation_ids_and_fit_normalization_do_not_leak(self):
        split = np.array(['train']*10 + ['test_random']*2)
        fit_ids, val_ids = split_training_ids(split, seed=7, validation_fraction=.2)
        self.assertEqual(len(fit_ids), 8)
        self.assertEqual(len(val_ids), 2)
        self.assertFalse(set(fit_ids) & set(val_ids))
        self.assertEqual(set(fit_ids) | set(val_ids), set(range(10)))
        repeated = split_training_ids(split, seed=7, validation_fraction=.2)
        np.testing.assert_array_equal(repeated[0], fit_ids)
        outputs = np.arange(12*2*3, dtype=float).reshape(12, 2, 3)
        mean, scale = fit_normalization(outputs, fit_ids)
        outputs[val_ids] = 1e6
        outputs[10:] = -1e6
        changed_mean, changed_scale = fit_normalization(outputs, fit_ids)
        np.testing.assert_array_equal(changed_mean, mean)
        np.testing.assert_array_equal(changed_scale, scale)


if __name__ == '__main__':
    unittest.main()
