Loaded best checkpoint from epoch: 19
Test loss: 0.4305
Test metrics:
dice: 0.9268
iou: 0.9203
precision: 0.1417
recall: 0.1387
global_precision: 0.8366
global_recall: 0.9043
Test positive/negative-separated metrics:
positive_slice_dice: 0.8973540869378226
positive_slice_precision: 0.9126249279688906
positive_slice_recall: 0.893413997027609
negative_slice_mean_fp_pixels: 3.0881552076242342
negative_slice_fp_sample_rate: 0.06773315180394826
num_positive_slices: 540
num_negative_slices: 2938
Test lesion-level metrics:
sensitivity: 0.9438596480188776
precision: 0.7082601048257819
fp_per_slice: 0.09545715928694652
gt_total: 855
pred_total: 1138
fp_lesions_total: 332
Test boundary metrics:
hd95_pos_mean_px: 8.04703633642394
hd95_pos_median_px: 1.0
hd95_pos_count: 517
hd95_gt_no_pred_count: 23

## Compare to epoch 10

#### Epoch 8 model was more sensitive:

- detects slightly more CAC lesions,
- recovers more true CAC pixels
- has slightly better positive-slice Dice
  But:
- produces more fp lesion blobs
- has lower global precision
- has more fp pixels on negative slices

#### Epoch 19 model was more selective:

- fewer fp pixels
- fewer fp lesions
- better lesion precision
- better global precision
- better mean HD95
  But:
- misses a few more true lesions/slices,
- pos Dice and Recall drop slightly.

## Questions for model's epoch choice: 8 or 19

Epoch 8 has strong pos-slice Dice / Recall and Lesion Sensitivity
Epoch 19 doesn't get hurt by visible false-positive specks.
