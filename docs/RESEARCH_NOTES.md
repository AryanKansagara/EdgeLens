# Research questions and reproducibility

1. Does weight-only INT8 representation change held-out digit classification?
2. Can a portable C++ runtime reproduce the quantized Python reference?
3. What is the parameter-storage tradeoff, separate from runtime speed?
4. Does the image preprocessing path behave consistently after lossless video I/O?

The scikit-learn digit dataset is deliberately small, so this is an accessible
end-to-end experiment rather than novel research. Strong classical baselines are
included to avoid implying that a neural network must be the best choice.

Training recipe, data split and seed are fixed in source. The model is selected by
validation accuracy, not final test results. Comparisons report both accuracy and
prediction agreement; matching class labels alone would hide logit differences.
Timing is local and excludes preprocessing. Model storage excludes runtime overhead.

Relevant primary references:
- [scikit-learn digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
- [PyTorch Linear layers](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html)
- [OpenCV documentation](https://docs.opencv.org/4.x/)
- [FFmpeg documentation](https://ffmpeg.org/documentation.html)

The CNN extension in `cnn_experiment.py` adds a two-convolution classifier on the
same digit split; see `CNN_REPORT.md` for its separate results and limitations.

Future work: a trained lightweight convolutional model on a larger, appropriately
licensed image dataset; writer/source-disjoint evaluation; representative natural
video; operator-level profiling; and actual target-device measurements using a
reviewed deployment backend. Android/JNI and diffusion-based augmentation should
be separate, tested extensions rather than résumé claims without implementation.
