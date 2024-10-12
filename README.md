# Segmentation Models for Phase-Contrast Microscopy

This repository contains 5 deep learning models for image segmentation, trained on 480 binary annotation images from the [OSF phase-contrast microscopy dataset](https://osf.io/ysaq2/). The models include Attention U-Net, Mask R-CNN, Segformer, Swin U-Net, and TransResUnet. Each model comes with pre-trained weights and a complete pipeline for data augmentation, training, validation, and inference.

## Models and Performance

The following table gives the average precision calculated as:

$$
\text{Average Precision}(t) = \frac{TP(t)}{TP(t) + FP(t) + FN(t)}
$$

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
├── Convolution/
│   ├── Attention-U-Net/
│   │   ├── Attention-U-Net.ipynb
│   │   └── requirements.txt
│   ├── Mask-R-CNN/
│   │   ├── Mask-R-CNN.ipynb
│   │   └── requirements.txt
└── Transformer/
    ├── Segformer/
    │   ├── Segformer-training.py
    │   └── requirements.txt
    ├── Swin-U-Net/
    │   ├── swin-unet-inference.ipynb
    │   ├── swin-unet-training.ipynb
    │   └── requirements.txt
    └── TransResUnet/
        ├── Transresunet-training.ipynb
        └── requirements.txt
```
## Output of Segmentation Models

In the image below, the output from the segmentation models is presented. 

![All Binary Images](https://github.com/user-attachments/assets/fd9712f6-740f-47e7-b1e5-9af6acf9ab61)


> **Figure:** A visual comparison of the output from different convolutional and transformer-based models. The models are tasked with segmenting cellular structures in microscopy images. This comparison allows for a detailed assessment of how each model performs in complex areas where cell clumps and edges are hard to define.

---

### Summary of Visual Analysis

- **Attention U-Net**: Performs well in detecting distinct shapes and small objects with minimal noise. However, in densely packed areas, it sometimes merges adjacent cells into larger blobs.
- **Mask R-CNN**: Has issues in accurately delineating borders between clumped cells. It tends to produce broader and less precise boundaries.
- **Swin U-Net**: Capable of identifying smaller objects effectively but struggles with smooth border delineation, resulting in noisier output.
- **Segformer MiT-B0 & B3**: Show enhanced refinement at cell borders but still have difficulties with highly detailed regions, especially in denser areas.
---

### Out-of-training Distribution Inference

In following figure, five out-of-training distribution images were gathered from datasets to assess generalizability. 

![Dataset01](https://github.com/user-attachments/assets/80b12531-d0fe-4383-bd98-6a09ea52ea56)

> **Figure:** Comparison of binary output masks from Attention U-Net and TransResUNet-F with outof-
training distribution data for different densities in Dataset 01 - 090303. The left column
shows the original images, the middle column shows the output of Attention U-Net, and
the right column shows the output o1f 0T3ransResUNet-F.

### Summary of Out-of-training Distribution Inference

- **Generalization**: Both Attention U-Net and TransResUnet-F were evaluated on out-of-training distribution data.
- **Segmentation Quality**: TransResUnet-F showed superior performance, especially in segmenting closely interconnected and overlapping cells, while Attention U-Net had difficulties, often introducing holes in the segmentation.
- **Cell Density Representation**: TransResUnet-F captured cell density more accurately, including rounded cell bodies that were missed by Attention U-Net.
- **Structural Fidelity**: TransResUnet-F better preserved the structure in densely packed regions, showing higher inter-connectivity between cells compared to the smoother, less accurate output from Attention U-Net.
- **Noise and Artifacts**: Attention U-Net struggled with border detection and introduced noise, particularly in oval-shaped clusters, which were better handled by TransResUnet-F.

---

