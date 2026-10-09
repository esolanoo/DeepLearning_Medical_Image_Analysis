# Deep Learning Model Comparison for Medical Image Analysis

## Overview

This project evaluates and compares three deep learning architectures for medical image analysis using the **Kvasir-SEG gastrointestinal polyp dataset**.

The objective is to study the trade-offs between predictive performance, generalization, model complexity, and computational cost under a consistent experimental protocol.

The selected architectures represent two convolutional neural network (CNN) approaches and one Vision Transformer (ViT).

## Objectives

* Compare three pretrained deep learning architectures.
* Evaluate predictive performance using appropriate classification metrics.
* Analyze generalization and potential overfitting.
* Measure training time, inference time, parameter count, and GPU memory consumption when available.
* Compare convergence speed and training stability.
* Provide reproducible experiments and documented implementation details.

## Dataset

**Kvasir-SEG** contains 1,000 gastrointestinal endoscopy images with corresponding pixel-level polyp segmentation masks.

* Domain: Gastroenterology
* Modality: RGB colonoscopy images
* Annotations: Polyp segmentation masks
* Original task: Semantic segmentation
* Dataset source: https://datasets.simula.no/kvasir-seg/

**Important:** All original Kvasir-SEG images depict polyps. Binary polyp-versus-normal classification therefore requires additional negative images. The final classification labels and their source must be documented before training.

## Models

| Model              | Architecture | Purpose                                               |
| ------------------ | ------------ | ----------------------------------------------------- |
| ResNet-18          | CNN          | Baseline performance                                  |
| EfficientNet-B0    | CNN          | Accuracy–efficiency trade-off                         |
| Vision Transformer | Transformer  | Comparison with attention-based image representations |

All models will use pretrained weights and the same training, validation, and test partitions wherever applicable.

## Methodology

1. Review the literature on medical image classification and polyp analysis.
2. Define the classification task and label-generation procedure.
3. Inspect the dataset and establish reproducible data splits.
4. Resize and normalize images using the appropriate pretrained model's input requirements.
5. Apply training-only data augmentation.
6. Fine-tune the three architectures using a common experimental protocol.
7. Evaluate predictive performance, generalization, convergence, and computational cost.
8. Compare results and discuss limitations.
9. Prepare the final scientific article in IEEE conference format.

## Evaluation Metrics

### Classification performance

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

ROC-AUC may also be reported if both classes are available and the test protocol supports it.

### Computational performance

* Total and trainable parameters
* Training time per epoch and total training time
* Inference latency
* Peak GPU memory usage, when available
* Epoch at which the best validation score is achieved

### Generalization

* Training versus validation performance
* Validation loss and learning curves
* Final performance on an untouched test set
* Variability across repeated runs, if computational resources permit

## Reproducibility

The repository will include:

* Fixed random seeds
* Documented dataset splits
* Explicit preprocessing and augmentation settings
* Model configurations and hyperparameters
* Training and evaluation scripts
* Saved metrics and experiment configurations
* Instructions for installing dependencies and reproducing results

## Repository Structure

See the project directory structure below.

## Requirements

* Python 3.10 or compatible environment
* PyTorch
* torchvision
* NumPy
* pandas
* scikit-learn
* Pillow
* Matplotlib
* Seaborn
* tqdm
* psutil

GPU acceleration is recommended but not mandatory.

## Results

Results will be added after all experiments have been completed. No performance values are assumed in advance.

## Limitations

This study uses a small, specialized medical image dataset. Results may depend on the selected label-generation procedure, pretrained weights, random seed, and hardware. Performance on Kvasir-SEG alone does not establish clinical validity or broad generalization to other hospitals or endoscopy systems.

## References

Jha, D. et al., “Kvasir-SEG: A Segmented Polyp Dataset,” in *MultiMedia Modeling*, 2020.

Official dataset: https://datasets.simula.no/kvasir-seg/

## License and Data Usage

Follow the dataset's official terms of use. Cite the original dataset publication in any report or publication using Kvasir-SEG.
