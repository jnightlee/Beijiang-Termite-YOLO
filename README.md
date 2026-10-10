# Beijiang-Termite-YOLO

Supporting materials for the paper:

> Li, H.; Wang, Y.; Liu, D.; Wang, X.; Fan, J.; Yu, M. Research and Application of an Intelligent Monitoring System for Levee Termite Activity Based on Multi-Optical-Path Infrared Triggering and YOLOv11. *Applied Sciences* (under review).

The repository contains the annotated image dataset, the training configuration and logs, the trained weights, and a script that reproduces the detection metrics reported in Table 3 of the paper.

## Contents

| Path | Description |
|---|---|
| `datasets/termite/images/` | 163 images acquired between November 2025 and April 2026 by nine monitoring stations in the Chihuaji termite source area of the Beijiang Levee, Guangdong, China, which is separate from the Shangdou area where the 268 stations of the field trial were installed: 76 images (1280 × 720 px) from six TBS stations of the same model as those used in the trial (station IDs `YYCHJ-T2-*`) and 87 images (1600 × 1200 px) from three monitoring stations of another type used in the same project (station IDs `YYCHJ-T3-*`). No images from the 268 trial stations are included. File names follow `<station ID>__<date>__<timestamp>.jpg`. |
| `datasets/termite/images/*/*.json` | Original annotations (LabelMe 4.0.0, rectangular bounding boxes). |
| `datasets/termite/labels/` | Annotations converted to YOLO format. |
| `datasets/termite/data.yaml` | Dataset configuration (5 classes). |
| `runs/termite_yolo11s_img1280/` | **Model reported in the paper.** Training arguments (`args.yaml`), training log (`results.csv`), curves, confusion matrices and weights (`weights/best.pt`). Trained with Ultralytics 8.3.169 in May 2026. |
| `runs/termite_test/` | Preliminary run (YOLO11n, 640 px, 30 epochs); not used in the paper. |
| `weights/` | COCO-pretrained `yolo11n.pt` and `yolo11s.pt` used for initialization. |
| `evaluate.py` | Reproduces the per-class metrics in Table 3 and the parameter count and GFLOPs of the model. |

## Dataset

Classes: `termite`, `mud_tube`, `red_imported_fire_ant`, `common_ant`, `ant_egg`. No red imported fire ants occur in the current images.

| Class | Training (images / instances) | Validation (images / instances) |
|---|---|---|
| termite | 2 / 39 | 1 / 3 |
| mud_tube | 25 / 28 | 31 / 52 |
| red_imported_fire_ant | 0 / 0 | 0 / 0 |
| common_ant | 59 / 752 | 18 / 1275 |
| ant_egg | 8 / 15 | 13 / 373 |
| Background images | 28 | 2 |
| Total images | 112 | 51 |

The images from the `YYCHJ-T3-*` stations, whose monitoring chamber differs from that of the TBS, were used only for training; they contain common ants, ant eggs and background scenes. All termite and mud-tube instances and all validation images come from the TBS stations. The split was made at the image level, so images from one station appear in both sets and the two sets overlap in time. No independent test set is available; the validation metrics should therefore be regarded as preliminary.

## Training configuration

YOLO11s (unmodified), COCO-pretrained; input size 1280 px; batch size 8; up to 200 epochs with early stopping (patience 60); optimizer selected automatically by the framework (AdamW, initial learning rate ≈1.1 × 10⁻³, weight decay 5 × 10⁻⁴), 3-epoch warm-up and cosine schedule; augmentation: HSV jitter, translation, scaling, horizontal flipping, mosaic (closed for the last 10 epochs), RandAugment and random erasing. Full settings are in `runs/termite_yolo11s_img1280/args.yaml`. Training was performed on an NVIDIA GeForce RTX 4060 GPU.

## Reproducing Table 3

```bash
pip install ultralytics==8.3.169
python evaluate.py              # CPU
python evaluate.py --device 0   # GPU
```

The script evaluates `weights/best.pt` on the validation set with a confidence threshold of 0.001 and an IoU threshold of 0.7, and prints precision, recall, F1-score, mAP@0.5 and mAP@0.5:0.95 for each class.

## Data not included

Station coordinates and the complete platform records of the field trial are not included because they concern the safety of an operating flood-control levee. They are available from the corresponding author on reasonable request, subject to permission from the Beijiang River Basin Administration of Guangdong Province.

## Citation

Please cite the paper above if you use these materials. Full bibliographic details will be added after publication.

## License

All rights reserved; available for academic use with citation.
