# Deep Learning for Gastrointestinal Image Classification

## Overview
This project benchmarks deep learning models for eight-class gastrointestinal image classification using the [Kvasir dataset](https://datasets.simula.no/kvasir/). It compares predictive performance and computational efficiency across pretrained models and a CNN trained from scratch.

## Models
- **ResNet-18** — CNN with residual connections.
- **EfficientNet-B0** — computationally efficient CNN.
- **ViT-B/16** — Vision Transformer.
- **AlexNet** — CNN baseline.
- **Small CNN** — trained from scratch.

EfficientNet-B0 is also evaluated using different training strategies: pretrained feature extraction, training from scratch, and fine-tuning.

## Dataset
The dataset contains eight classes: dyed-lifted polyps, dyed-resection margins, esophagitis, normal cecum, normal pylorus, normal Z-line, polyps, and ulcerative colitis.

## How to Run (Windows CMD)

1. Clone the repository and enter its directory:
   ```cmd
   git clone https://github.com/esolanoo/DeepLearning_Medical_Image_Analysis
   cd <REPOSITORY_FOLDER>
   ```

2. Create and activate a virtual environment:
   ```cmd
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```cmd
   pip install -r requirements.txt
   ```

4. Configure the dataset path and experiment settings as required by the scripts.

5. Run the training or evaluation script:
   ```cmd
   python train.py
   ```

Replace the placeholders with the actual folder name, and script names. Check the scripts for any additional arguments or configuration requirements.

## Evaluation
Models are compared using accuracy, macro-F1, per-class metrics, Matthews correlation coefficient (MCC), model parameters, training time, and inference latency.
