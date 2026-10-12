| Setting                      | Initial choice                              |
| ---------------------------- | ------------------------------------------- |
| Task                         | Multi-class polyp classification            |
| Dataset                      | Kvasir, (8 classes)*1,000 images            |
| Split                        | 80% train, 10% validation, 10% test         |
| Random seed                  | 5338                                        |
| Input resolution             | 224 × 224                                   |
| Image format                 | RGB                                         |
| Image normalization          | ImageNet mean and standard deviation        |
| Training augmentation        | Horizontal flips, small rotations, jitter on train |
| Validation/test augmentation | None                                        |
| Initial loss                 | CrossEntropyLoss                            |

### Structure

1. Introduction — Medical image classification, limited labeled data, research questions, and contributions.
2. Related Work — Transfer learning in medical imaging, CNNs, vision transformers, and interpretability.
3. Materials and Methods — Kvasir-v2, preprocessing, data splits, architectures, pretrained versus random initialization, optimization, and metrics.
4. Experimental Setup — Hardware, seeds, early stopping, learning-rate scheduling, and data-subset experiments.
5. Results — Main test results, learning curves, data-efficiency curves, per-class metrics, confusion matrices, and computational cost.
6. Discussion — Why pretraining helps or does not, architecture trade-offs, difficult classes, and interpretability findings.
7. Limitations and Future Work — Single dataset, limited external validation, domain shift, self-supervised learning, multimodal clinical data, and external testing.
8. Conclusion — Answer the research questions directly.

### Proposed research questions

**How much does ImageNet pretraining improve gastrointestinal image classification when labeled training data are limited, and how do architecture, model complexity, and interpretability affect the results?**

RQ1 — Does pretraining help with limited labeled data?
Compare pretrained vs randomly initialized weights vs pretrained-finetuned

RQ2 - What trade-offs exist between predictive performance and computational cost?
Compare macro-F1, inference latency, parameter counts, training time, and convergence across the four architectures.

RQ3 - How do model errors and visual explanations differ across architectures?
Examine per-class errors and use Grad-CAM for CNNs and suitable attention visualization for ViT-B/16. Treat these visualizations as diagnostic evidence, not proof that the model's reasoning is clinically valid.

### Dataset


### Methodology 
| Setting              | Recommended approach                                                   |
| -------------------- | ---------------------------------------------------------------------- |
| Dataset split        | 5,600 / 1,200 / 1,200 split                                            |
| Input resolution     | 224 × 224 for all models                                               |
| Batch size           | Start at 16 or 32; reduce if CUDA runs out of memory                   |
| Pretrained models    | 10 epochs maximum, early stopping                                      |
| Scratch CNN          | Up to 50 epochs, early stopping                                        |
| Checkpoint selection | Best validation macro-F1                                               |
| Test set             | Evaluate only after the experimental decisions are finalized           |
| Randomness           | Fix seeds and record them                                              |
| Reporting            | Record epochs completed, training time, parameter count, and GPU model |
| Evaluation           | Accuracy, macro_f1, macro_specificity, matthews correlation coeficient, inference_seconds, ms_per_image, validation_macro_f1 |

### Models

| model                     | model_description                                                                                                      | total_parameters         | trainable_parameters               |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ------------------------ | ---------------------------------- |
| resnet18                  | Pre-trained CNN. Residual connections help optimize deeper networks                                                    |            11,180,616    |                              4,104 |
| efficientnet_b0           | Pre-trained CNN. Balances network depth, width, and input resolution to improve efficiency.                            |               4,017,796  |                            10,248  |
| vit_b_16                  | Pre-trained Transformer. A straightforward global-attention baseline                                                   |            85,804,808    |                              6,152 |
| alexnet                   | Historical CNN baseline                                                                                                |            57,036,616    |                            32,776  |
| small_cnn                 | Locally-built, light-weight CNN.                                                                                       |                  390,952 |                          390,952   |
| efficientnet_b0_scratch   | EfficientNet with randomly assigned weights and backbone freeze                                                        |               4,017,796  |                      4,017,796     |
| efficientnet_b0_finetuned | Fine-tuned from already-trained pretrained  \`efficientnet_b0\` checkpoint, sequential continuation of that experiment |               4,017,796  |                      4,017,796     |



### Train results

| model                     | total_parameters | trainable_parameters | best_epoch | best_val_macro_f1 | epochs_completed | training_seconds |
| ------------------------- | ---------------- | -------------------- | ---------- | ----------------- | ---------------- | ---------------- |
| efficientnet_b8_finetuned | 4017796          | 4017796              | 25         | 0.914842          | 25               | 3940             |
| vit_b_16                  | 85804808         | 6152                 | 7          | 0.902163          | 10               | 2474             |
| efficientnet_b0           | 4017796          | 10248                | 9          | 0.890066          | 10               | 1223             |
| efficientnet_b0_scratch   | 4017796          | 4017796              | 38         | 0.883605          | 43               | 6663             |
| resnet18                  | 11180616         | 4104                 | 10         | 0.868076          | 10               | 1297             |
| small_cnn                 | 390952           | 390952               | 33         | 0.839722          | 38               | 4991             |
| alexnet                   | 57036616         | 32776                | 8          | 0.804682          | 10               | 1258             |

### Initial Findings

Pretraining appears valuable. ViT exceeds the scratch CNN by about 17.37 percentage points in validation macro-F1. This supports the hypothesis that pretrained representations are useful when the available labeled dataset is relatively small.

ViT-B/16 has the strongest overall predictive performance. Its macro-F1 is about 1.36 percentage points above EfficientNet-B0 and 4.14 points above ResNet-18, but also the largest model. It has about 85.8 million parameters, compared with 391,000 for the scratch CNN. The scratch CNN has roughly 219 times fewer parameters. Its backbone is frozen, so this is a pretrained feature-extraction result, not full fine-tuning.

EfficientNet-B0 is a promising practical compromise. EfficientNet-B0 offers a promising performance-speed trade-off. It is only about 1.34 percentage points behind ViT in macro-F1, with roughly 35% lower measured inference time per image, with about 4 million parameters.

SmallCNN is a credible lightweight baseline. It achieves almost the same macro-F1 as ResNet-18, despite having far fewer parameters. However, their difference is tiny and it trained 20 more epochs

note: for a defensible comparison, use ImageNet-pretrained AlexNet with only its final classification layer trained, consistent with your other pretrained models. Reaches 0.7829 before early stopping. It is a historical architecture, but its 57M parameters make it relatively large for this performance. Since it ran for fewer epochs, treat its result as an initial baseline rather than a definitive ranking.

### Model Evaluation

| model                   | test_samples | accuracy | macro_f1 | macro_specificity | mcc      | inference_seconds | ms_per_image | checkpoint_epoch | validation_macro_f1 |
| ----------------------- | ------------ | -------- | -------- | ----------------- | -------- | ----------------- | ------------ | ---------------- | ------------------- |
| vit_b_16                | 1200         | 0.901667 | 0.901615 | 0.985952          | 0.887702 | 38.20587          | 31.83823     | 7                | 0.902163            |
| efficientnet_b0         | 1200         | 0.888333 | 0.888015 | 0.984048          | 0.872722 | 24.17701          | 20.14751     | 9                | 0.890066            |
| efficientnet_b0_scratch | 1200         | 0.8725   | 0.872261 | 0.981786          | 0.854328 | 21.3232           | 17.76933     | 38               | 0.883605            |
| small_cnn               | 1200         | 0.860833 | 0.860174 | 0.980119          | 0.841718 | 20.17101          | 16.80918     | 33               | 0.839722            |
| resnet18                | 1200         | 0.86     | 0.85979  | 0.98              | 0.840592 | 20.03087          | 16.69239     | 10               | 0.868076            |
| alexnet                 | 1200         | 0.8225   | 0.821075 | 0.974643          | 0.797675 | 24.47368          | 20.39473     | 8                | 0.804682            |