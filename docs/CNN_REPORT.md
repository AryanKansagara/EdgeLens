# CNN experiment: training, regularization and evaluation

Executed on 2026-10-27 on macOS ARM64 CPU with PyTorch 2.10.0. This extends the small digit-image benchmark, not the mobile-device deployment path.

Architecture: Conv2d(1,16,3,padding=1), ReLU, Conv2d(16,32,3,padding=1), ReLU, MaxPool2d(2), Flatten, Dropout(0.2), Linear(512,10). Total parameters: 9,930.

Recipe fixed in source: seed 23, 40 epochs, AdamW learning rate 0.001, weight decay 0.01, cross-entropy loss, batch size 64. Seeded stratified splits contain 1,077 training, 270 validation and 450 test images. Checkpoint selection uses minimum validation cross-entropy, with epoch 40 selected in this run. No training or validation augmentation is implemented.

| Evaluation set | Accuracy | Cross-entropy |
| --- | ---: | ---: |
| Training | 99.91% | 0.01466 |
| Validation | 98.15% | 0.08154 |
| Test, clean | 97.78% | 0.05637 |
| Test, additive Gaussian noise sigma 0.15 | 95.56% | 0.11918 |

Images use intensities in [0,1]; the noise probe uses seed 99 and clips to that interval. It is evaluated after checkpoint selection and is not used to tune the model. Raw metrics and validation-loss history are stored in `results/cnn_metrics.json`. All 27 tests passed, including six CNN-specific checks for layout, shapes, finite input, evaluation-mode dropout and checkpoint restoration.

Interpretation: the classifier performs well on this small digit split and loses accuracy under this particular corruption. One seed, one corruption level and one reused test split do not establish broad generalization. The split is not writer-disjoint. Dropout and weight decay are implemented but no ablation demonstrates their causal effect. The pre-existing PCA+SVM baseline reached 98.89%, so deep learning is not automatically the best method here.

This CNN has no INT8 export, C++ execution, mobile-device benchmark or Android/JNI integration. Storage and C++ parity results elsewhere belong to the separate MLP. No claim of state-of-the-art vision, natural-video understanding, industry deployment or real-world camera robustness is made. Code was developed with AI assistance and should be reviewed before use.
