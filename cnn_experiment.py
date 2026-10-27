"""Small CNN extension; independent from the MLP quantization/C++ path."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from vision import SEED, split_data


class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
                                    nn.MaxPool2d(2), nn.Flatten(), nn.Dropout(0.2),
                                    nn.Linear(32 * 4 * 4, 10))

    def forward(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != (1, 8, 8):
            raise ValueError('expected N x 1 x 8 x 8 images')
        return self.layers(x)


def image_tensor(values):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 64 or not np.isfinite(values).all():
        raise ValueError('expected finite N x 64 image matrix')
    return torch.from_numpy(values.reshape(-1, 1, 8, 8))


def score(model, x, y):
    model.eval()
    with torch.inference_mode():
        logits = model(x)
        return {'accuracy': float((logits.argmax(1) == y).float().mean()),
                'cross_entropy': float(nn.functional.cross_entropy(logits, y))}


def fit(x, y, vx, vy, epochs=40):
    torch.manual_seed(SEED)
    model = DigitCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    best, state, selected, history = float('inf'), None, 0, []
    for epoch in range(epochs):
        model.train()
        for batch in torch.randperm(len(x)).split(64):
            loss = nn.functional.cross_entropy(model(x[batch]), y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        validation = score(model, vx, vy)
        history.append({'epoch': epoch + 1, **validation})
        if validation['cross_entropy'] < best:
            best, selected = validation['cross_entropy'], epoch + 1
            state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(state)
    model.eval()
    return model, selected, history


def experiment(output):
    torch.set_num_threads(1)
    values, labels, train, validation, test = split_data()
    images, targets = image_tensor(values), torch.from_numpy(labels)
    model, selected, history = fit(images[train], targets[train], images[validation], targets[validation])
    # Fixed corruption probe, not used for training or checkpoint selection.
    rng = np.random.default_rng(99)
    noisy = np.clip(values[test] + rng.normal(0, 0.15, values[test].shape), 0, 1)
    metrics = {'architecture': 'two Conv2d layers, ReLU, MaxPool, Dropout, Linear',
               'train_images': len(train), 'validation_images': len(validation), 'test_images': len(test),
               'seed': SEED, 'epochs': 40, 'selected_epoch': selected,
               'dropout': 0.2, 'adamw_weight_decay': 0.01,
               'train': score(model, images[train], targets[train]),
               'validation': score(model, images[validation], targets[validation]),
               'test': score(model, images[test], targets[test]),
               'gaussian_noise_sigma_0_15': score(model, image_tensor(noisy), targets[test]),
               'parameters': sum(parameter.numel() for parameter in model.parameters()),
               'history': history,
               'limitations': ['One seed and small digits dataset; no writer-disjoint split or mobile deployment.',
                               'CNN is not quantized or run in C++; those results belong to the separate MLP path.',
                               'No regularization ablation; no claim dropout caused any accuracy improvement.',
                               'Existing test split reused for this exploratory extension; not a new untouched benchmark.']}
    output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / 'cnn.pt')
    (output / 'cnn_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps({key: value for key, value in metrics.items() if key != 'history'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('artifacts'))
    experiment(parser.parse_args().output)
