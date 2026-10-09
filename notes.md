| Setting                      | Initial choice                              |
| ---------------------------- | ------------------------------------------- |
| Task                         | Multi-class polyp segmentation              |
| Dataset                      | Kvasir, 8*1,000 images                      |
| Split                        | 75% train, 15% validation, 15% test         |
| Random seed                  | 5338                                        |
| Input resolution             | 256 × 256                                   |
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