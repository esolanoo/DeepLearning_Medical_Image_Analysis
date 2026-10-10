| Setting                      | Initial choice                              |
| ---------------------------- | ------------------------------------------- |
| Task                         | Multi-class polyp segmentation              |
| Dataset                      | Kvasir, 8*1,000 images                      |
| Split                        | 80% train, 10% validation, 10% test         |
| Random seed                  | 5338                                        |
| Input resolution             | 224 × 224                                   |
| Image format                 | RGB                                         |
| Image normalization          | ImageNet mean and standard deviation        |
| Training augmentation        | Horizontal flips, small rotations, jitter on train |
| Validation/test augmentation | None                                        |
| Initial loss                 | Binary cross-entropy plus Dice loss         |
| Model selection              | Best validation Dice                        |
| Final evaluation             | Held-out test set                           |


### Proposed research questions

Which architecture achieves the best accuracy and macro-F1 on the Kvasir classification task?

Which model generalizes best, based on the gap between training and validation performance and its held-out test results?

Which architecture offers the best trade-off between predictive performance and computational cost?

How quickly does each model converge under the same training protocol?


### First run
| Model               | Best validation macro-F1 | Training time | Total parameters | Trainable parameters |
| ------------------- | ------------------------ | ------------- | ---------------- | -------------------- |
| ViT-B/16            | 0.9046                   | 21.5 min      | 85.8M            | 6,152                |
| EfficientNet-B0     | 0.8798                   | 17.3 min      | 4.0M             | 10,248               |
| ResNet-18           | 0.8626                   | 17.5 min      | 11.2M            | 4,104                |
| AlexNet             | 0.7829                   | 8.6 min*      | 57.0M            | 32,776               |
| Small CNN (scratch) | 0.7309                   | 17.3 min      | 391K             | 390,952              |

### Initial Findings

Pretraining appears valuable. ViT exceeds the scratch CNN by about 17.37 percentage points in validation macro-F1. This supports the hypothesis that pretrained representations are useful when the available labeled dataset is relatively small.

ViT has the best validation score, but also the largest model. It has about 85.8 million parameters, compared with 391,000 for the scratch CNN. The scratch CNN has roughly 219 times fewer parameters. Its backbone is frozen, so this is a pretrained feature-extraction result, not full fine-tuning.

EfficientNet-B0 is a promising practical compromise. It achieves 87.98% macro-F1, only 2.48 percentage points below ViT, with about 4 million parameters.

The scratch CNN is not a failure. It achieves 73.09% macro-F1 without pretrained weights. It simply has not matched the pretrained models under this training protocol. Ten epochs may not be enough for it to converge.

One caution: total training time is similar for all four models because even frozen pretrained backbones still perform forward passes through the entire network. The scratch CNN's smaller parameter count does not automatically translate into faster training

note: for a defensible comparison, use ImageNet-pretrained AlexNet with only its final classification layer trained, consistent with your other pretrained models. Reaches 0.7829 before early stopping. It is a historical architecture, but its 57M parameters make it relatively large for this performance. Since it ran for fewer epochs, treat its result as an initial baseline rather than a definitive ranking.


====================================================================
Evaluating small_cnn
Accuracy:           0.7483
Macro-F1:           0.7447
Macro-specificity:  0.9640
MCC:                0.7160
Inference time:     141.02 s (117.52 ms/image)