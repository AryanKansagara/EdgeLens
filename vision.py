"""Reproducible image classification, quantization and constructed-video analysis."""
import argparse
import json
import platform
from pathlib import Path
import struct
import subprocess

import cv2
import imageio_ffmpeg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import torch
from torch import nn

SEED = 23


class ImageMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))

    def forward(self, values):
        return self.layers(values)


def split_data():
    data = load_digits()
    values = (data.data / 16).astype(np.float32)
    labels = data.target.astype(np.int64)
    train, test = train_test_split(np.arange(len(labels)), test_size=0.25, stratify=labels, random_state=SEED)
    train, validation = train_test_split(train, test_size=0.20, stratify=labels[train], random_state=SEED)
    return values, labels, train, validation, test


def quantize(weights, bias):
    weights, bias = np.asarray(weights, dtype=np.float32), np.asarray(bias, dtype=np.float32)
    if weights.ndim != 2 or bias.shape != (weights.shape[0],) or not weights.size:
        raise ValueError('invalid layer dimensions')
    if not np.isfinite(weights).all() or not np.isfinite(bias).all():
        raise ValueError('layer values must be finite')
    maxima = np.max(np.abs(weights), axis=1)
    scales = np.where(maxima == 0, 1.0, maxima / 127).astype(np.float32)
    codes = np.clip(np.rint(weights / scales[:, None]), -127, 127).astype(np.int8)
    return codes, scales, bias.copy()


def quantized_layers(model):
    return [quantize(layer.weight.detach().numpy(), layer.bias.detach().numpy())
            for layer in model.layers if isinstance(layer, nn.Linear)]


def infer_numpy(values, layers):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('expected a finite sample matrix')
    for index, (codes, scales, bias) in enumerate(layers):
        if values.shape[1] != codes.shape[1]:
            raise ValueError('sample/layer shape mismatch')
        values = (values @ codes.astype(np.float32).T) * scales + bias
        if index + 1 < len(layers):
            values = np.maximum(values, 0)
    return values


def write_model(path, layers):
    with Path(path).open('wb') as stream:
        stream.write(b'ELQ1' + struct.pack('<I', len(layers)))
        for codes, scales, bias in layers:
            stream.write(struct.pack('<II', *codes.shape))
            stream.write(codes.tobytes(order='C'))
            stream.write(scales.astype('<f4').tobytes())
            stream.write(bias.astype('<f4').tobytes())


def write_samples(path, values):
    with Path(path).open('wb') as stream:
        stream.write(b'ELS1' + struct.pack('<II', *values.shape))
        stream.write(values.astype('<f4').tobytes())


def preprocess(frame):
    if frame is None or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError('expected a uint8 BGR frame')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return (cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA).reshape(1, 64) / 255).astype(np.float32)


def video_analysis(output, values, labels, model, layers):
    # Construct a clip from held-out images. This is not natural video tracking.
    frames = []
    truth = []
    for image, label in zip(values[:30], labels[:30]):
        pixels = np.rint(image.reshape(8, 8) * 255).astype(np.uint8)
        frame = cv2.cvtColor(cv2.resize(pixels, (128, 128), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
        frames.extend([frame] * 4)
        truth.extend([int(label)] * 4)
    raw = np.stack(frames)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    clip = output/'constructed_digits.mkv'
    command = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '128x128', '-r', '10', '-i', 'pipe:0', '-c:v', 'ffv1', str(clip)]
    subprocess.run(command, input=raw.tobytes(), capture_output=True, check=True, timeout=30)
    capture = cv2.VideoCapture(str(clip))
    predictions, baseline_predictions, annotated = [], [], []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        sample = preprocess(frame)
        prediction = int(infer_numpy(sample, layers).argmax(axis=1)[0])
        with torch.inference_mode():
            baseline = int(model(torch.from_numpy(sample)).argmax(dim=1)[0])
        predictions.append(prediction); baseline_predictions.append(baseline)
        view = cv2.copyMakeBorder(frame, 0, 32, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
        cv2.putText(view, f'FP32 {baseline} / INT8 {prediction}', (4, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (240, 240, 240), 1)
        annotated.append(view)
    capture.release()
    if len(predictions) != len(truth):
        raise RuntimeError('decoded frame count mismatch')
    command = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '128x160', '-r', '10', '-i', 'pipe:0', '-c:v', 'ffv1', str(output/'annotated_digits.mkv')]
    subprocess.run(command, input=np.stack(annotated).tobytes(), capture_output=True, check=True, timeout=30)
    return {'frames': len(predictions), 'unique_source_images': 30,
            'constructed_clip_accuracy': float(accuracy_score(truth, predictions)),
            'fp32_int8_prediction_agreement': float(np.mean(np.array(predictions) == baseline_predictions)),
            'warning': '120 repeated frames from 30 held-out still images; not independent natural video samples.'}


def experiment(output, epochs=35):
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1); torch.manual_seed(SEED); np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    values, labels, train, validation, test = split_data()
    model = ImageMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    loss_function = nn.CrossEntropyLoss()
    train_x, train_y = torch.from_numpy(values[train]), torch.from_numpy(labels[train])
    best_score, best_state, best_epoch = -1, None, 0
    history = []
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(len(train))
        for offset in range(0, len(train), 128):
            indices = permutation[offset:offset+128]
            optimizer.zero_grad(); loss = loss_function(model(train_x[indices]), train_y[indices]); loss.backward(); optimizer.step()
        model.eval()
        with torch.inference_mode():
            score = float((model(torch.from_numpy(values[validation])).argmax(dim=1).numpy() == labels[validation]).mean())
        history.append(score)
        if score > best_score:
            best_score, best_epoch = score, epoch + 1
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()
    with torch.inference_mode():
        logits = model(torch.from_numpy(values[test])).numpy()
    layers = quantized_layers(model)
    quantized_logits = infer_numpy(values[test], layers)
    baseline_results = {}
    for name, estimator in (
        ('logistic_regression', make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))),
        ('pca_svm', make_pipeline(StandardScaler(), PCA(n_components=32, random_state=SEED), SVC(kernel='rbf', C=5))),
    ):
        estimator.fit(values[train], labels[train])
        baseline_results[name] = float(accuracy_score(labels[test], estimator.predict(values[test])))
    write_model(output/'model.elq', layers); write_samples(output/'test.els', values[test])
    executable = output/'edge_runtime'
    root = Path(__file__).resolve().parent
    subprocess.run(['clang++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Wpedantic', str(root/'runtime.cpp'), '-o', str(executable)], check=True, timeout=120)
    result = subprocess.run([str(executable), str(output/'model.elq'), str(output/'test.els')], text=True, capture_output=True, check=True, timeout=30)
    cpp = json.loads(result.stdout)
    difference = float(np.max(np.abs(np.array(cpp['logits']) - quantized_logits)))
    if difference > 0.0005 or not np.array_equal(cpp['predictions'], quantized_logits.argmax(axis=1)):
        raise RuntimeError('C++/NumPy quantized inference parity failed')
    # Compile and run the same real-model path with memory/undefined-behavior checks.
    sanitized = output/'edge_runtime_sanitized'
    subprocess.run(['clang++', '-std=c++17', '-g', '-fsanitize=address,undefined', '-fno-omit-frame-pointer', str(root/'runtime.cpp'), '-o', str(sanitized)], check=True, timeout=120)
    subprocess.run([str(sanitized), str(output/'model.elq'), str(output/'test.els')], capture_output=True, check=True, timeout=30)
    fp32_bytes = sum(layer.weight.numel()*4 + layer.bias.numel()*4 for layer in model.layers if isinstance(layer, nn.Linear))
    int8_bytes = sum(codes.nbytes + scales.nbytes + bias.nbytes for codes, scales, bias in layers)
    metrics = {
        'dataset': 'scikit-learn load_digits: 1797 real 8x8 handwritten-digit images',
        'seed': SEED, 'train_images': len(train), 'validation_images': len(validation), 'test_images': len(test),
        'epochs': epochs, 'selected_epoch_by_validation': best_epoch, 'best_validation_accuracy': best_score,
        'pytorch_fp32_test_accuracy': float(accuracy_score(labels[test], logits.argmax(axis=1))),
        'int8_test_accuracy': float(accuracy_score(labels[test], quantized_logits.argmax(axis=1))),
        'fp32_int8_prediction_agreement': float(np.mean(logits.argmax(axis=1) == quantized_logits.argmax(axis=1))),
        'classical_baselines_test_accuracy': baseline_results,
        'fp32_parameter_bytes': fp32_bytes, 'int8_parameter_and_scale_bytes': int8_bytes,
        'parameter_storage_reduction_percent': 100*(1-int8_bytes/fp32_bytes),
        'quantized_model_file_bytes': (output/'model.elq').stat().st_size,
        'cpp_numpy_max_logit_difference': difference,
        'cpp_numpy_prediction_agreement': float(np.mean(cpp['predictions'] == quantized_logits.argmax(axis=1))),
        'cpp_batch_amortized_ms_per_image': cpp['batch_amortized_ms_per_image'],
        'cpu_platform': platform.platform(), 'inference_device': 'local CPU, not Android or an edge board',
        'video': video_analysis(output, values[test], labels[test], model, layers),
        'limitations': ['Small digit benchmark, not a state-of-the-art claim.', 'Weight-only INT8; activations/accumulation remain FP32.', 'Model storage is not process RSS or peak inference memory.', 'Latency is a local batch-amortized measurement, not mobile-device latency.'],
    }
    (output/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    matrix = confusion_matrix(labels[test], quantized_logits.argmax(axis=1), labels=np.arange(10))
    fig, axis = plt.subplots(figsize=(7, 6)); graphic = axis.imshow(matrix, cmap='Blues')
    axis.set(xticks=range(10), yticks=range(10), xlabel='Predicted class', ylabel='True class', title='Held-out INT8 digit classification (450 images)')
    for i in range(10):
        for j in range(10):
            axis.text(j, i, str(matrix[i,j]), ha='center', va='center', color='white' if matrix[i,j] > matrix.max()/2 else 'black', fontsize=8)
    fig.colorbar(graphic, ax=axis, label='Image count'); fig.tight_layout(); fig.savefig(output/'confusion.png', dpi=140); plt.close(fig)
    print(json.dumps(metrics, indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts'))
    parser.add_argument('--epochs', type=int, default=35)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 100:
        parser.error('epochs must be 1..100')
    experiment(args.output.resolve(), args.epochs)


if __name__ == '__main__':
    main()
