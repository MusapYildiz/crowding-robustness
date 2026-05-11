"""
prepare_dataset.py

COIL-100 veri setinden isolation görseller üretir.
- Siyah background threshold ile maskelenir
- Aspect ratio korunarak resize edilir
- Uniform siyah canvas'a yerleştirilir
- Instance bazlı strict split (train/val/test)

Kullanım:
    python data/prepare_dataset.py --config configs/config.yaml
"""

import os, json, random, argparse
import cv2
import numpy as np
from pathlib import Path
import yaml
from tqdm import tqdm


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def remove_dark_bg(img_bgr: np.ndarray, threshold: int = 40) -> np.ndarray:
    """Koyu background piksellerini maskeler."""
    gray   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    kernel  = np.ones((5, 5), np.uint8)
    mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask    = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    return mask


def crop_to_object(img_bgr: np.ndarray,
                   mask: np.ndarray) -> tuple:
    """Maskeye göre nesnenin bounding box'ını kırpar."""
    coords = cv2.findNonZero(mask)
    if coords is None:
        return img_bgr, mask
    x, y, w, h = cv2.boundingRect(coords)
    return img_bgr[y:y+h, x:x+w], mask[y:y+h, x:x+w]


def place_on_canvas(img_bgr: np.ndarray, mask: np.ndarray,
                    object_size: int, canvas_size: int,
                    bg_color: int) -> np.ndarray:
    """
    Aspect ratio koruyarak resize + sadece maske pikselleri canvas'a koy.
    """
    h, w   = img_bgr.shape[:2]
    scale  = object_size / max(h, w)
    new_w  = max(int(w * scale), 1)
    new_h  = max(int(h * scale), 1)

    resized_obj  = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(mask,    (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    canvas   = np.full((canvas_size, canvas_size, 3), bg_color, dtype=np.uint8)
    offset_y = (canvas_size - new_h) // 2
    offset_x = (canvas_size - new_w) // 2

    roi = canvas[offset_y:offset_y+new_h, offset_x:offset_x+new_w]
    roi[resized_mask > 0] = resized_obj[resized_mask > 0]
    canvas[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = roi

    return canvas


def main(config_path: str):
    cfg         = load_config(config_path)
    coil_dir    = Path(cfg["data"]["coil_dir"])
    output_dir  = Path(cfg["data"]["output_dir"])
    canvas_size = cfg["image"]["canvas_size"]
    object_size = cfg["image"]["object_size"]
    bg_color    = cfg["image"]["background_color"]
    threshold   = cfg["image"]["bg_threshold"]
    train_ratio = cfg["data"]["split"]["train_ratio"]
    val_ratio   = cfg["data"]["split"]["val_ratio"]
    seed        = cfg["data"]["split"]["random_seed"]

    target_ids   = cfg["data"]["target_obj_ids"]
    external_ids = cfg["data"]["external_obj_ids"]
    all_ids      = target_ids + external_ids

    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_train, manifest_val, manifest_test = {}, {}, {}

    for obj_id in all_ids:
        angles = list(range(0, 360, 5))  # 72 açı
        rng.shuffle(angles)

        n       = len(angles)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)

        splits = [
            ("train", angles[:n_train]),
            ("val",   angles[n_train:n_train+n_val]),
            ("test",  angles[n_train+n_val:]),
        ]

        for split_name, split_angles in splits:
            split_dir = output_dir / split_name / f"obj{obj_id}"
            split_dir.mkdir(parents=True, exist_ok=True)

            for angle in tqdm(split_angles,
                              desc=f"{split_name}/obj{obj_id}", leave=False):
                img_path = coil_dir / f"obj{obj_id}__{angle}.png"
                if not img_path.exists():
                    continue

                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                mask                    = remove_dark_bg(img, threshold)
                cropped_img, cropped_mask = crop_to_object(img, mask)
                canvas                  = place_on_canvas(
                    cropped_img, cropped_mask,
                    object_size, canvas_size, bg_color
                )

                key      = f"{obj_id}_{angle}"
                out_path = split_dir / f"{key}.png"
                cv2.imwrite(str(out_path), canvas)

                entry = {
                    "obj_id":    obj_id,
                    "angle":     angle,
                    "is_target": obj_id in target_ids,
                }
                if split_name == "train":   manifest_train[key] = entry
                elif split_name == "val":   manifest_val[key]   = entry
                else:                       manifest_test[key]  = entry

        print(f"obj{obj_id}: train={len([a for a in angles[:int(n*train_ratio)]])}"
              f" val={n_val} test={n - int(n*train_ratio) - n_val}")

    # Leakage kontrol
    for split_name, split_manifest in [("val", manifest_val),
                                        ("test", manifest_test)]:
        overlap = set(manifest_train.keys()) & set(split_manifest.keys())
        if overlap:
            raise RuntimeError(f"LEAKAGE: train ∩ {split_name} = {len(overlap)} key")
        print(f"[OK] train ∩ {split_name} = ∅")

    # Manifest kaydet
    for name, data in [("train", manifest_train),
                       ("val",   manifest_val),
                       ("test",  manifest_test)]:
        out = output_dir / f"{name}_manifest.json"
        with open(out, "w") as f:
            json.dump(data, f, indent=2)

    print(f"\nTamamlandı:")
    print(f"  train: {len(manifest_train)}")
    print(f"  val:   {len(manifest_val)}")
    print(f"  test:  {len(manifest_test)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
