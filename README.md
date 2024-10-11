# Segmentation Models for Phase-Contrast Microscopy

This repository contains 5 deep learning models for image segmentation, trained on 480 binary annotation images from the [OSF phase-contrast microscopy dataset](https://osf.io/ysaq2/). The models include Attention U-Net, Mask R-CNN, Segformer, Swin U-Net, and TransResUnet. Each model comes with pre-trained weights and a complete pipeline for data augmentation, training, validation, and inference.

## Models and Performance

| Threshold | Mask R-CNN (40 epochs) | Attention U-Net (Focal Loss) | MIT-B0 | MIT-B3 | MIT-B5 | Swin U-Net (no transforms) | TransResUnet (raw) | TransResUnet (Finetuned - 1967 samples) |
| --------- | ---------------------- | ---------------------------- | ------ | ------ | ------ | -------------------------- | ------------------ | --------------------------------------- |
| **0.5**   | 0.8230                 | 0.8425                       | 0.5313 | 0.5623 | 0.5739 | 0.4051                     | 0.6563            | 0.7245                                 |
| **0.6**   | 0.7577                 | 0.7925                       | 0.4755 | 0.5065 | 0.5174 | 0.3447                     | 0.6146            | 0.6845                                 |
| **0.7**   | 0.6119                 | 0.6794                       | 0.3948 | 0.4331 | 0.4369 | 0.2637                     | 0.5463            | 0.6230                                 |
| **0.8**   | 0.2769                 | 0.3969                       | 0.2122 | 0.2491 | 0.2441 | 0.1162                     | 0.3786            | 0.4738                                 |
| **0.9**   | 0.0156                 | 0.0460                       | 0.0206 | 0.0238 | 0.0265 | 0.0090                     | 0.0473            | 0.0775                                 |


## Features

- **Data Pipeline**: The code includes a complete data pipeline for loading, augmenting, and preprocessing the dataset.
- **Data Augmentation**: Various geometric and pixel-wise augmentations are applied to enhance model performance.
- **Loss Functions**: Several loss functions, including Dice, BCE, and Focal Loss, are implemented to handle class imbalance.
- **Training & Validation**: Models are trained and validated with average precision metrics, and validation scores are logged after each epoch.
- **Inference**: Code for running inference on new images is provided for each model.

## Repository Structure

```bash
Segmentation-Models-Phase-Contrast-Microscopy/
├── Attention-U-Net/
│   ├── model/
│   ├── weights/
│   ├── Attention-U-Net.ipynb
│   └── README.md
├── Mask-R-CNN/
│   ├── model/
│   ├── weights/
│   └── Mask-R-CNN.ipynb
├── Segformer/
│   ├── model/
│   ├── weights/
│   └── Segformer.ipynb
├── Swin-U-Net/
│   ├── model/
│   ├── weights/
│   └── Swin-U-Net.ipynb
└── TransResUnet/
    ├── model/
    ├── weights/
    └── TransResUnet.ipynb
