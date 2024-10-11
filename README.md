# Segmentation Models for Phase-Contrast Microscopy

This repository contains 5 deep learning models for image segmentation, trained on 480 binary annotation images from the [OSF phase-contrast microscopy dataset](https://osf.io/ysaq2/). The models include Attention U-Net, Mask R-CNN, Segformer, Swin U-Net, and TransResUnet. Each model comes with pre-trained weights and a complete pipeline for data augmentation, training, validation, and inference.

## Models and Performance

| Model         | Average Precision (AP) | Threshold |
| ------------- | ---------------------- | --------- |
| **Attention U-Net**  | 0.8425                   | 0.5       |
| **Mask R-CNN**       | 0.823                    | 0.5       |
| **TransResUnet**     | 0.7244                   | 0.5       |
| **Segformer**        | -                        | -         |
| **Swin U-Net**       | -                        | -         |

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
