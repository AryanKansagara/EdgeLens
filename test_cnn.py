import unittest

import numpy as np
import torch

from cnn_experiment import DigitCNN, fit, image_tensor, score


class CNNTests(unittest.TestCase):
    def test_shape(self):
        self.assertEqual(tuple(DigitCNN()(torch.zeros(3, 1, 8, 8)).shape), (3, 10))

    def test_invalid_shape(self):
        with self.assertRaises(ValueError):
            DigitCNN()(torch.zeros(3, 64))

    def test_finite_input(self):
        with self.assertRaises(ValueError):
            image_tensor(np.full((3, 64), np.nan))

    def test_image_layout(self):
        values = np.arange(128).reshape(2, 64)
        np.testing.assert_array_equal(image_tensor(values).numpy().reshape(2, 64), values)

    def test_evaluation_disables_dropout(self):
        model = DigitCNN().eval()
        x = torch.rand(3, 1, 8, 8)
        torch.testing.assert_close(model(x), model(x))

    def test_train_and_restore(self):
        torch.set_num_threads(1)
        torch.manual_seed(20)
        x, y = torch.rand(12, 1, 8, 8), torch.arange(12) % 10
        model, selected, history = fit(x, y, x[:4], y[:4], epochs=2)
        self.assertIn(selected, (1, 2))
        self.assertEqual(len(history), 2)
        restored = DigitCNN().eval()
        restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(model(x), restored(x))
        self.assertTrue(np.isfinite(score(model, x, y)['cross_entropy']))


if __name__ == '__main__':
    unittest.main()
