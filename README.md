# Deep Learning for Gastrointestinal Image Classification

## Overview

This project compares three deep learning architectures for multiclass gastrointestinal image classification using the **Kvasir dataset**. The objective is to evaluate how convolutional neural networks (CNNs) and Vision Transformers perform when classifying endoscopic images into eight categories.

## Models

* **ResNet-18:** CNN baseline with residual connections.
* **EfficientNet-B0:** computationally efficient CNN.
* **Vision Transformer (ViT-B/16):** transformer-based image classifier.

All models will use ImageNet-pretrained weights and a consistent experimental protocol.

## Dataset

The [Kvasir dataset](https://datasets.simula.no/kvasir/) contains endoscopic images representing anatomical landmarks, pathological findings, and endoscopic procedures:

* Dyed-lifted polyps
* Dyed-resection margins
* Esophagitis
* Normal cecum
* Normal pylorus
* Normal Z-line
* Polyps
* Ulcerative colitis

## Methodology

1. Inspect the dataset and verify class distributions.
2. Resize and normalize images; apply data augmentation to training data only.
3. Create stratified training, validation, and test splits.
4. Train and evaluate the three pretrained architectures.
5. Compare classification performance, generalization, convergence, and computational cost.

## Evaluation

* Accuracy and macro-F1 score
* Per-class precision, recall, and F1-score
* Training and inference time
* Number of parameters and peak GPU memory
* Training and validation learning curves

## Reproducibility

The repository will include preprocessing scripts, model training and evaluation code, configuration files, random seeds, and saved experimental results.

## Scope

This is an academic image-classification experiment, not a clinically validated diagnostic system.
