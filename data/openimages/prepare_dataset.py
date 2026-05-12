"""
data/openimages/prepare_dataset.py

refined_masks_manifest.json'dan isolation gorseller uretir.
Her instance:
  - SAM2 maskesiyle arka plandan temizlenir
  - Uzun kenar object_long_edge px'e scale edilir (aspect ratio korunur)
  - 224x224 siyah canvas'in merkezine yerlestirilir
  - Instance bazli strict split uygulanir (train/val/test)

Kullanim:
    python data/openimages/prepare_dataset.py --config configs/config_openimages.yaml
"""

import json, random, argparse
import numpy as np
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def is_valid_instance(image: np.ndarray, mask: np.ndarray,
                       bbox: list, min_mask_ratio: float,
                       min_color_std: float) -> bool:
    """Kalitesiz instance'lari filtreler."""
    x1, y1, x2, y2 = bbox
    h_img, w_img   = image.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w_img, x2); y2 = min(h_img, y2)

    if x2 <= x1 or y2 <= y1:
        return False

    crop_rgb  = image[y1:y2, x1:x2]
    crop_mask = mask[y1:y2, x1:x2]
    bbox_area = (x2 - x1) * (y2 - y1)

    if bbox_area == 0:
        return False

    # Maske doluluk orani
    if crop_mask.sum() / (bbox_area * 255) < min_mask_ratio:
        return False

    # Renk cesitliligi
    masked_px = crop_rgb[crop_mask > 127]
    if len(masked_px) == 0 or masked_px.std() < min_color_std:
        return False

    return True


def extract_and_place(image: np.ndarray, mask: np.ndarray,
                       bbox: list, long_edge: int,
                       canvas_size: int, bg_color: int) -> np.ndarray | None:
    """
    Maskeyle nesneyi keser, aspect ratio koruyarak
    long_edge px'e scale eder, canvas merkezine yerlestirir.
    """
    x1, y1, x2, y2 = bbox
    h_img, w_img   = image.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w_img, x2); y2 = min(h_img, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    crop_rgb  = image[y1:y2, x1:x2]
    crop_mask = mask[y1:y2, x1:x2]

    if crop_rgb.size == 0 or crop_mask.sum() == 0:
        return None

    # Aspect ratio koruyarak resize
    ch, cw = crop_rgb.shape[:2]
    scale  = long_edge / max(ch, cw)
    new_w  = max(int(cw * scale), 1)
    new_h  = max(int(ch * scale), 1)

    resized_obj  = cv2.resize(crop_rgb,  (new_w, new_h),
                               interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(crop_mask, (new_w, new_h),
                               interpolation=cv2.INTER_NEAREST)

    # Canvas
    canvas   = np.full((canvas_size, canvas_size, 3),
                        bg_color, dtype=np.uint8)
    offset_y = (canvas_size - new_h) // 2
    offset_x = (canvas_size - new_w) // 2

    roi = canvas[offset_y:offset_y+new_h, offset_x:offset_x+new_w]
    roi[resized_mask > 127] = resized_obj[resized_mask > 127]
    canvas[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = roi

    return canvas


def main(config_path: str):
    cfg = load_config(config_path)

    output_dir  = Path(cfg["data"]["output_dir"])
    manifest_path = output_dir / "refined_masks_manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest bulunamadi: {manifest_path}\n"
            "Once refine_masks.py calistirin."
        )

    with open(manifest_path) as f:
        manifest = json.load(f)

    target_classes  = cfg["data"]["target_classes"]
    max_per_class   = cfg["data"]["max_samples_per_class"]
    canvas_size     = cfg["image"]["canvas_size"]
    long_edge       = cfg["image"]["object_long_edge"]
    bg_color        = cfg["image"]["background_color"]
    min_mask_ratio  = cfg["image"]["min_mask_ratio"]
    min_color_std   = cfg["image"]["min_color_std"]
    train_ratio     = cfg["data"]["split"]["train_ratio"]
    val_ratio       = cfg["data"]["split"]["val_ratio"]
    seed            = cfg["data"]["split"]["random_seed"]

    rng = random.Random(seed)

    # Sinif bazli instance'lari grupla
    class_instances = {c: [] for c in target_classes}
    for ann_id, info in manifest.items():
        label = info["label"]
        if label in class_instances:
            class_instances[label].append((ann_id, info))

    manifest_train, manifest_val, manifest_test = {}, {}, {}

    print("Instance'lar isleniyor...")
    for label in target_classes:
        instances = class_instances[label]
        rng.shuffle(instances)
        instances = instances[:max_per_class]

        n       = len(instances)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)

        splits = [
            ("train", instances[:n_train]),
            ("val",   instances[n_train:n_train+n_val]),
            ("test",  instances[n_train+n_val:]),
        ]

        for split_name, split_instances in splits:
            split_dir = output_dir / split_name / label
            split_dir.mkdir(parents=True, exist_ok=True)

            saved = 0
            for ann_id, info in tqdm(split_instances,
                                      desc=f"  {split_name}/{label}",
                                      leave=False):
                image = cv2.imread(info["image_path"])
                mask  = cv2.imread(info["mask_path"],
                                    cv2.IMREAD_GRAYSCALE)

                if image is None or mask is None:
                    continue

                bbox = info["bbox"]  # [x1, y1, x2, y2]

                if not is_valid_instance(image, mask, bbox,
                                          min_mask_ratio, min_color_std):
                    continue

                canvas = extract_and_place(
                    image, mask, bbox,
                    long_edge, canvas_size, bg_color
                )
                if canvas is None:
                    continue

                out_path = split_dir / f"{ann_id}.png"
                cv2.imwrite(str(out_path), canvas)
                saved += 1

                entry = {
                    "label":      label,
                    "image_path": info["image_path"],
                    "bbox":       bbox,
                }
                if split_name == "train":   manifest_train[ann_id] = entry
                elif split_name == "val":   manifest_val[ann_id]   = entry
                else:                       manifest_test[ann_id]  = entry

            print(f"    {split_name}/{label}: {saved} gorsel")

    # Leakage kontrol
    for split_name, split_m in [("val", manifest_val),
                                  ("test", manifest_test)]:
        overlap = set(manifest_train) & set(split_m)
        if overlap:
            raise RuntimeError(
                f"LEAKAGE: train n {split_name} = {len(overlap)}"
            )
        print(f"[OK] train n {split_name} = 0")

    # Manifest kaydet
    for name, data in [("train", manifest_train),
                        ("val",   manifest_val),
                        ("test",  manifest_test)]:
        out = output_dir / f"{name}_manifest.json"
        with open(out, "w") as f:
            json.dump(data, f, indent=2)

    print(f"\nTamamlandi:")
    print(f"  train: {len(manifest_train)}")
    print(f"  val:   {len(manifest_val)}")
    print(f"  test:  {len(manifest_test)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_openimages.yaml")
    args = parser.parse_args()
    main(args.config)
