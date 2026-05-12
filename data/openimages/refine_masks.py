"""
data/openimages/refine_masks.py

SAM2 modeli ile Open Images segmentasyon maskelerini iyilestirir.
Her instance icin:
  1. Mevcut maskeyi bounding box'a donustur
  2. SAM2'ye bbox prompt olarak ver
  3. Iyilestirilmis maskeni kaydet

Kullanim:
    python data/openimages/refine_masks.py --config configs/config_openimages.yaml

Gereksinimler:
    pip install sam2
    # veya: pip install git+https://github.com/facebookresearch/sam2.git
"""

import os, json, argparse
import numpy as np
from pathlib import Path

import cv2
import yaml
import torch


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_sam2():
    """SAM2 modelini yukler."""
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError:
        raise ImportError(
            "SAM2 gerekli:\n"
            "pip install git+https://github.com/facebookresearch/sam2.git"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"SAM2 yukleniyor... (device: {device})")

    predictor = SAM2ImagePredictor.from_pretrained(
        "facebook/sam2-hiera-large"
    )
    return predictor


def refine_mask_with_sam2(predictor, image_rgb: np.ndarray,
                           bbox: list) -> np.ndarray:
    """
    Verilen bbox'i SAM2'ye prompt olarak verip
    iyilestirilmis binary mask dondurur.

    Args:
        predictor  : SAM2ImagePredictor
        image_rgb  : (H, W, 3) uint8
        bbox       : [x1, y1, x2, y2] piksel koordinatlari
    Returns:
        binary_mask: (H, W) uint8, 0/255
    """
    predictor.set_image(image_rgb)

    input_box = np.array(bbox, dtype=np.float32)
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )

    # En iyi maske
    best_mask = masks[np.argmax(scores)]
    return (best_mask * 255).astype(np.uint8)


def process_fiftyone_dataset(cfg: dict, predictor):
    """
    fiftyone dataset'indeki her instance icin SAM2 ile maske iyilestirir.
    Sonuclari output_dir/refined_masks/ altina kaydeder.
    """
    try:
        import fiftyone as fo
    except ImportError:
        raise ImportError("fiftyone gerekli: pip install fiftyone")

    output_dir  = Path(cfg["data"]["output_dir"])
    masks_dir   = output_dir / "refined_masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    target_classes = cfg["data"]["target_classes"]
    max_per_class  = cfg["data"]["max_samples_per_class"]

    # Mevcut dataset'i yukle
    try:
        dataset = fo.load_dataset("crowding_openimages")
    except Exception:
        raise RuntimeError(
            "fiftyone dataset bulunamadi. "
            "Once download.py calistirin."
        )

    print(f"Toplam sample: {len(dataset)}")

    manifest = {}
    class_counts = {c: 0 for c in target_classes}

    for sample in dataset.iter_samples(progress=True):
        if sample.ground_truth is None:
            continue

        # Gorseli yukle
        image_bgr = cv2.imread(sample.filepath)
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        for det in sample.ground_truth.detections:
            label = det.label
            if label not in target_classes:
                continue
            if class_counts[label] >= max_per_class:
                continue

            # fiftyone bbox: [x1, y1, w, h] normalized
            bx, by, bw, bh = det.bounding_box
            x1 = int(bx * w)
            y1 = int(by * h)
            x2 = int((bx + bw) * w)
            y2 = int((by + bh) * h)

            # Gecerlilik kontrolu
            if x2 <= x1 or y2 <= y1:
                continue
            if (x2 - x1) * (y2 - y1) < 400:  # min 20x20
                continue

            # SAM2 ile maske iyilestir
            try:
                refined_mask = refine_mask_with_sam2(
                    predictor, image_rgb, [x1, y1, x2, y2]
                )
            except Exception as e:
                print(f"  [WARN] SAM2 hatasi {sample.id}: {e}")
                continue

            # Kaydet
            ann_id   = f"{sample.id}_{det.id}"
            out_path = masks_dir / f"{ann_id}.png"
            cv2.imwrite(str(out_path), refined_mask)

            manifest[ann_id] = {
                "image_path":  sample.filepath,
                "label":       label,
                "bbox":        [x1, y1, x2, y2],
                "mask_path":   str(out_path),
                "sample_id":   sample.id,
            }

            class_counts[label] += 1

    # Manifest kaydet
    out = output_dir / "refined_masks_manifest.json"
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nIyilestirilen instance sayisi: {len(manifest)}")
    print("\nSinif bazli:")
    for cls in target_classes:
        print(f"  {cls}: {class_counts[cls]}")

    return manifest


def main(config_path: str):
    cfg       = load_config(config_path)
    predictor = load_sam2()
    process_fiftyone_dataset(cfg, predictor)
    print("\nTamamlandi.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_openimages.yaml")
    args = parser.parse_args()
    main(args.config)
