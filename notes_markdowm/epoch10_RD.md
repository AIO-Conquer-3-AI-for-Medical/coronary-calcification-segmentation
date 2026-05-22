## Raw stats

Loaded best checkpoint from epoch: 8
Test loss: 0.4298
Test metrics:
dice: 0.9123
iou: 0.9065
global_precision: 0.8003
global_recall: 0.9293
Test positive/negative-separated metrics:
positive_slice_dice: 0.9053282577973618
positive_slice_precision: 0.9151163935799289
positive_slice_recall: 0.9097633189073315
negative_slice_mean_fp_pixels: 3.9118447923757658
negative_slice_fp_sample_rate: 0.08645336963921035
num_positive_slices: 540
num_negative_slices: 2938
Test lesion-level metrics:
sensitivity: 0.9578947357217605
precision: 0.6554160120444144
fp_per_slice: 0.12622196664749857
gt_total: 855
pred_total: 1274
fp_lesions_total: 439
Test boundary metrics:
hd95_pos_mean_px: 9.587876085647052
hd95_pos_median_px: 1.0
hd95_pos_count: 521
hd95_gt_no_pred_count: 19

## Summary / keypoints

| Metric                          | Result | Interpretation                                                      |
| ------------------------------- | ------ | ------------------------------------------------------------------- |
| Positive-slice Dice             | 0.9053 | Very strong mask overlap on CAC-positive slices                     |
| Positive-slice Precision        | 0.9151 | Predicted CAC pixels on positive slices are mostly correct          |
| Positive-slice Recall           | 0.9098 | Recovers most CAC pixels on positive slices                         |
| Global pixel Precision          | 0.8003 | Across the entire test set, 80% of predicted CAC pixels are correct |
| Global pixel Recall             | 0.9293 | Across the entire test set, it recovers 93% of true CAC pixels      |
| Lesion sensitivity              | 0.9579 | Dtects nearly 96% of GT lesion components                           |
| Lesion precision                | 0.6554 | About 65.5% of predicted lesion blobs correspond to GT lesions      |
| Negative FP slice rate          | 8.65%  | 1 in 12 negative slices gets any fp CAC                             |
| Mean fp pixels / negative slice | 3.91   | Fp on negative slices are usually small                             |
| hd95_pos_median_px              | 1.0    | >= half pred boundary is extr close to GT boundary                  |

# Insights

Conclusions:

- segments CAC regions accurately on positive slices
- detects most individual CAC lesions
- keeps fp relatively controlled (``)
