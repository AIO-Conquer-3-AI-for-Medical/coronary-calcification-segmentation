The 2.5D U-Net has learned the core task: segment CAC moderately well on test cases, with good lesion detection and substantially controlled false positives

| Metric | Result | Judgment |
| Positive-slice Dice | 0.8400 | Strong |
| Positive-slice Precision | 0.8574 | Strong |
| Positive-slice Recall | 0.8371 | Strong |
| Global pixel Precision | 0.6456 | Decent, room to improve |
| Global pixel Recall | 0.8040 | Good |
| Negative FP slice rate | 9.30% | Much improved, still relevant |
| Lesion sensitivity | 0.8785 | Strong |
| Lesion precision | 0.6079 | Moderate |
| FP lesions total | 567 | Still the main weakness |

## A. On slices have CAC, segmentation is good

Refer to `positive_slice_dice/precision/recall`. The model is learning true CAC shape / location, not just bright-pixel noise.

## B. Globally, the model finds most CAC pixels but still creates some fp pixels

Refer to `global_precision/recall`. The model still over-segments enough that fp suppression remains a meaningful next target.

## Fp on negative slices

Refer to `negative_slice_fp_sample_rate`, `negative_slice_mean_fp_pixels`.

## Lesion-level results

Refer to `lesion sensitivity`, `lesion precision`. The model still generates a notable number of extra predicted lesion components.

## Validation and Test difference

The model generalizes, but the test set is clearly harder than validation, or validation was easier by split composition.

val_sep_metrics["positive_slice_dice"]
