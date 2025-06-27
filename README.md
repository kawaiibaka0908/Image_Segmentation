#  Optimized U-Net++ for Semantic Segmentation

[![View Dataset on Kaggle](https://img.shields.io/badge/Kaggle-View_Dataset-blue?logo=kaggle)](https://www.kaggle.com/competitions/iitg-ai-overnight-hackathon-2024)

This project implements an optimized **U-Net++ architecture** for semantic segmentation, designed for pixel-wise classification tasks. The model is trained on color-encoded masks and achieves strong accuracy and inference speed, making it suitable for applications such as autonomous driving and real-time object segmentation.

---

##  Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Details](#training-details)
- [Inference Pipeline](#inference-pipeline)
- [Results](#results)
- [Limitations](#limitations)
- [Future Work](#future-work)

---

##  Introduction

Semantic segmentation classifies each pixel in an image into predefined categories. U-Net++ improves over standard U-Net by using **nested skip connections**, which enhance feature reuse and gradient flow. This leads to better segmentation performance, especially in imbalanced datasets.

---

##  Dataset

-  **Source**: [IITG AI Overnight Hackathon 2024 – Kaggle Competition](https://www.kaggle.com/competitions/iitg-ai-overnight-hackathon-2024)
- **Input**: RGB images
- **Labels**: Color-encoded masks
- **Preprocessing**:
  - Resized all images and masks to `128x128` for memory optimization
  - Applied class-based color encoding for categorical segmentation
  - Designed a custom vectorized label-mapping function to handle class boundary pixels accurately

---

##  Model Architecture

Built using a custom U-Net++ architecture with the following components:

- **Encoder Path**:
  - cblock1: 32 filters
  - cblock2: 64 filters
  - cblock3: 128 filters + dropout
  - cblock4: 256 filters + dropout
  - cblock5: 512 filters

- **Decoder Path**:
  - ublock6: upsample 256 filters
  - ublock7: upsample 128 filters
  - ublock8: upsample 64 filters
  - ublock9: upsample 32 filters

- **Final Layer**: 1×1 convolution with softmax activation for multi-class output

- **Skip Connections**: Dense skip connections used for better information flow

---

##  Training Details

- **Loss Functions**: Dice Loss + Categorical Cross-Entropy
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Epochs**: Trained for 10, final weights saved after 2 epochs (Kaggle resource constraints)
- **Batch Size**: 16
- **Environment**: Trained on Kaggle GPU

---

##  Inference Pipeline

- Preprocessed images are passed to the trained model
- Model outputs pixel-wise class probabilities
- **Post-processing includes**:
  - Thresholding predictions
  - Morphological operations (dilation/erosion)
  - Mapping back predicted IDs to color encodings for visualization

---

##  Results

| Metric             | Value        |
|--------------------|--------------|
| Validation Accuracy| ~80%         |
| Model Size         | 395.6 MB     |
| Inference Time     | 230 ms/image |

> **Note**: Due to technical issues during training, the final evaluation CSV was not generated.

---

##  Limitations

- Model training was interrupted multiple times on Kaggle
- Final evaluation metrics could not be exported
- Some classes remain underrepresented in the dataset
- Inference speed (230 ms) could be too high for critical real-time applications

---

##  Future Work

- Implement SMOTE/ADASYN for synthetic class balancing
- Integrate pre-trained backbones (e.g., ResNet, EfficientNet)
- Explore DeepLabV3+, FCN, SegNet for architectural comparisons
- Add attention modules like CBAM or SE-blocks
- Use quantization/pruning to reduce model size and speed up inference
- Evaluate with additional metrics: IoU, F1-score, mean accuracy

---

