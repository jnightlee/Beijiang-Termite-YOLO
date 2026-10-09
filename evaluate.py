"""Reproduce the detection metrics reported in Table 3 of the paper.

Usage:
    pip install ultralytics==8.3.169
    python evaluate.py              # CPU
    python evaluate.py --device 0   # GPU
"""
import argparse
import tempfile
from pathlib import Path

import yaml
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops, get_num_params

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "runs/termite_yolo11s_img1280/weights/best.pt"))
    ap.add_argument("--data", default=str(ROOT / "datasets/termite/data.yaml"))
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    # Point the dataset path to the local copy of the repository.
    cfg = yaml.safe_load(open(a.data, encoding="utf-8"))
    cfg["path"] = str(Path(a.data).resolve().parent)
    tmp = Path(tempfile.mkdtemp()) / "data.yaml"
    yaml.safe_dump(cfg, open(tmp, "w", encoding="utf-8"), allow_unicode=True)

    model = YOLO(a.weights)
    r = model.val(data=str(tmp), imgsz=a.imgsz, batch=1, conf=0.001, iou=0.7,
                  device=a.device, plots=False, verbose=False)
    b = r.box
    print(f"\n{'Class':<24}{'P':>8}{'R':>8}{'F1':>8}{'mAP50':>8}{'mAP50-95':>10}")
    for i, c in enumerate(r.ap_class_index):
        p, rec = float(b.p[i]), float(b.r[i])
        f1 = 2 * p * rec / (p + rec) if p + rec > 0 else 0.0
        print(f"{model.names[int(c)]:<24}{p:8.3f}{rec:8.3f}{f1:8.3f}{b.ap50[i]:8.3f}{b.ap[i]:10.3f}")
    print(f"{'all (mean)':<24}{b.mp:8.3f}{b.mr:8.3f}{'':>8}{b.map50:8.3f}{b.map:10.3f}")

    fused = model.model.fuse()
    print(f"\nParameters: {get_num_params(fused):,}")
    print(f"GFLOPs at {a.imgsz} px: {get_flops(fused, a.imgsz):.1f}")


if __name__ == "__main__":
    main()
