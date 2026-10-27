# EdgeLens

An image/video-analysis and model-optimization benchmark using Python, PyTorch,
NumPy, scikit-learn, OpenCV, FFmpeg, Matplotlib and a C++17 inference runtime.
Trains a small neural network on real handwritten-digit images, compares classical
ML baselines and tests weight-only INT8 inference as an edge-oriented storage tradeoff.

## Run

Python 3.12 and Clang are required. A CPU run is sufficient. Exact direct dependency
versions are in `requirements.txt`; availability depends on your platform.

```sh
python -m venv .venv
.venv/bin/pip install -r requirements.txt
MPLCONFIGDIR=.mplconfig .venv/bin/python -m unittest -v
MPLCONFIGDIR=.mplconfig .venv/bin/python vision.py --output artifacts
```

The experiment trains the neural network, selects its checkpoint by validation
accuracy, evaluates held-out images, fits logistic-regression and PCA+SVM baselines,
exports an INT8 model and compares C++ inference with NumPy. It also compiles/runs
the real-model inference path under AddressSanitizer and UndefinedBehaviorSanitizer.

## Dataset and evaluation design

Uses scikit-learn's bundled [load_digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
dataset: 1,797 real 8×8 handwritten-digit images across ten classes. Original source:
[UCI optical recognition of handwritten digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits).
The repository does not redistribute raw dataset images. Dataset loading is local
to scikit-learn; no new dataset download is needed after dependency installation.

Seeded, stratified train/validation/test splits are disjoint. Training and PCA/
scaling fit only training data; the fixed neural-network recipe chooses a checkpoint
using validation data. The held-out test split has 450 images. This random split is
not a writer-disjoint or real-world distribution-shift benchmark. No hyperparameter
search or test-based model selection is performed by the script.

## Model optimization and C++ runtime

Architecture: 64 → 128 → 64 → 10 with ReLU hidden layers. Quantization stores each
output channel's weights as signed INT8 plus FP32 scales and biases. Activations,
accumulation and outputs remain FP32. The C++ runtime validates the binary format,
dimensions, finite values and positive scales before executing portable scalar loops.

Storage metrics count model parameters/scales, not process memory. Serialized-model
bytes are reported separately. A smaller representation is not a speedup claim.
The runtime reports the median of nine local batch-amortized timings after three
warm-up passes, excluding file loading and image preprocessing but including
per-image runtime vector allocations. This is not Android/device latency, energy,
peak memory or comparison with an optimized framework backend.

## Video path

FFmpeg constructs a lossless 120-frame clip from 30 held-out images repeated four
times. OpenCV decodes it, converts BGR to grayscale and resizes for classification;
an annotated clip shows FP32/INT8 predictions. Prediction text is added only after
classification, never used as model input. Repeated frames are not independent
samples. This is a constructed-video I/O test, not natural-scene video, object
detection, temporal modeling, optical flow or tracking.

## Outputs and limitations

`artifacts/` contains the binary model, test inputs, C++ executables, clips, metrics
and a confusion matrix. It is ignored by Git. Published `results/` contains only
metrics and the confusion matrix, not raw dataset images or video.

No mobile/edge hardware deployment, Android application, JNI bridge, TensorFlow,
Stable Diffusion, FLUX, ControlNet, Photoshop, distillation or pruning is implemented.
This small image-classification benchmark is not state of the art. It makes no
claims about autonomous-driving or general real-world vision performance.

The inactive CI template in `ci/github-actions.yml` can be activated by a repository
owner with workflow permissions. See [research notes](docs/RESEARCH_NOTES.md).
