import json
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
import numpy as np
from vision import ImageMLP, infer_numpy, preprocess, quantize, quantized_layers, split_data, write_model, write_samples


class VisionTests(unittest.TestCase):
    def test_dataset_split_is_disjoint(self):
        values, labels, train, validation, test = split_data()
        self.assertEqual(len(values), 1797); self.assertEqual(len(test), 450)
        self.assertFalse(set(train) & set(test)); self.assertFalse(set(validation) & set(test)); self.assertFalse(set(train) & set(validation))
        self.assertEqual(len(train) + len(validation) + len(test), len(values))

    def test_class_coverage_and_normalization(self):
        values, labels, train, validation, test = split_data()
        self.assertTrue(np.isfinite(values).all()); self.assertGreaterEqual(values.min(), 0); self.assertLessEqual(values.max(), 1)
        for indices in (train, validation, test): self.assertEqual(set(labels[indices]), set(range(10)))

    def test_zero_channel_quantization(self):
        codes, scales, bias = quantize(np.zeros((2, 4)), np.zeros(2))
        self.assertTrue((codes == 0).all()); self.assertTrue((scales > 0).all())

    def test_per_channel_scales(self):
        codes, scales, _ = quantize(np.array([[1, -1], [10, -10]]), np.zeros(2))
        np.testing.assert_allclose(scales[1], scales[0]*10)
        self.assertEqual(int(codes.min()), -127); self.assertEqual(int(codes.max()), 127)

    def test_invalid_layer_shapes(self):
        for weights, bias in ((np.zeros(4), np.zeros(1)), (np.zeros((2, 3)), np.zeros(3)), (np.zeros((0, 3)), np.zeros(0))):
            with self.subTest(shape=weights.shape), self.assertRaises(ValueError): quantize(weights, bias)

    def test_nonfinite_layers(self):
        for bad in (np.nan, np.inf):
            with self.subTest(bad=bad), self.assertRaises(ValueError): quantize(np.array([[bad]]), np.zeros(1))

    def test_reference_quantized_inference(self):
        layer = quantize(np.array([[1., -1.]]), np.array([0.5]))
        np.testing.assert_allclose(infer_numpy(np.array([[3., 1.]]), [layer]), [[2.5]], atol=1e-5)

    def test_hidden_relu(self):
        layers = [quantize(np.array([[-1.]]), np.array([0.])), quantize(np.array([[1.]]), np.array([2.]))]
        np.testing.assert_allclose(infer_numpy([[3.]], layers), [[2.]])

    def test_input_shape_and_finite_validation(self):
        layer = quantize(np.ones((1, 2)), np.zeros(1))
        for values in ([1, 2], [[1, 2, 3]], [[np.nan, 0]]):
            with self.subTest(values=values), self.assertRaises(ValueError): infer_numpy(values, [layer])

    def test_frame_preprocessing_contract(self):
        sample = preprocess(np.full((128, 128, 3), 255, dtype=np.uint8))
        self.assertEqual(sample.shape, (1, 64)); np.testing.assert_allclose(sample, 1.)

    def test_invalid_frames(self):
        for frame in (None, np.zeros((8, 8), dtype=np.uint8), np.zeros((8, 8, 3), dtype=np.float32)):
            with self.subTest(frame_type=type(frame)), self.assertRaises(ValueError): preprocess(frame)

    def test_architecture_export_dimensions(self):
        layers = quantized_layers(ImageMLP())
        self.assertEqual([layer[0].shape for layer in layers], [(128, 64), (64, 128), (10, 64)])


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='edgelens-runtime-test-')
        cls.root = Path(cls.temporary.name); cls.executable = cls.root/'runtime'
        subprocess.run(['clang++', '-std=c++17', '-g', '-fsanitize=address,undefined', str(Path(__file__).parent/'runtime.cpp'), '-o', str(cls.executable)], check=True, timeout=120)

    @classmethod
    def tearDownClass(cls): cls.temporary.cleanup()

    def setUp(self):
        self.model = self.root/'model.elq'; self.samples = self.root/'samples.els'
        self.layers = [quantize(np.array([[1., -1.], [-1., 1.]]), np.array([0., 0.]))]
        write_model(self.model, self.layers); write_samples(self.samples, np.array([[3., 1.]], dtype=np.float32))

    def run_runtime(self):
        return subprocess.run([str(self.executable), str(self.model), str(self.samples)], capture_output=True, text=True, timeout=30)

    def assert_rejected(self):
        result = self.run_runtime(); self.assertEqual(result.returncode, 1); self.assertIn('edge_runtime:', result.stderr)

    def test_python_cpp_numerical_parity(self):
        result = self.run_runtime(); self.assertEqual(result.returncode, 0, result.stderr)
        actual = json.loads(result.stdout); np.testing.assert_allclose(actual['logits'], [[2, -2]], atol=1e-5); self.assertEqual(actual['predictions'], [0])

    def test_bad_magic(self): self.model.write_bytes(b'NOPE'); self.assert_rejected()
    def test_truncated_weights(self): self.model.write_bytes(self.model.read_bytes()[:-3]); self.assert_rejected()
    def test_zero_layer_count(self): self.model.write_bytes(b'ELQ1'+struct.pack('<I', 0)); self.assert_rejected()
    def test_zero_scale(self):
        write_model(self.model, [(self.layers[0][0], np.zeros(2, dtype=np.float32), self.layers[0][2])]); self.assert_rejected()
    def test_trailing_model_bytes(self): self.model.write_bytes(self.model.read_bytes()+b'x'); self.assert_rejected()
    def test_wrong_sample_shape(self): write_samples(self.samples, np.ones((1, 3), dtype=np.float32)); self.assert_rejected()
    def test_nonfinite_sample(self): write_samples(self.samples, np.array([[np.nan, 1]], dtype=np.float32)); self.assert_rejected()
    def test_zero_samples(self): write_samples(self.samples, np.ones((0, 2), dtype=np.float32)); self.assert_rejected()


if __name__ == '__main__': unittest.main()
