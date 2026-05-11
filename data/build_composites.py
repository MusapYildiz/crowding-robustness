"""
build_composites.py

COIL-100 isolation görsellerinden crowding kompozitleri üretir.

- Sol ve sağ flanker: aynı instance (same_flanker_both_sides=true)
- Flanker her spacing seviyesinde sabit kalır (aynı instance)
- Leakage kontrolü: flanker pool train setinden seçilmez

Kullanım:
    python data/build_composites.py --config configs/config.yaml
"""

import json, random, argparse, shutil
import cv2
import numpy as np
from pathlib import Path
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_pool(dataset_dir: Path, split: str, obj_id: int) -> list:
    d = dataset_dir / split / f"obj{obj_id}"
    return sorted(d.glob("*.png")) if d.exists() else []


def validate_no_leakage(dataset_dir: Path, split: str):
    train_keys = set(json.load(open(dataset_dir / "train_manifest.json")).keys())
    split_keys = set(json.load(open(dataset_dir / f"{split}_manifest.json")).keys())
    overlap    = train_keys & split_keys
    if overlap:
        raise RuntimeError(
            f"[LEAKAGE] train ∩ {split} = {len(overlap)} — "
            f"örnek: {list(overlap)[:5]}"
        )
    print(f"  [OK] train ∩ {split} = ∅ ({len(split_keys)} instance)")


def select_flanker(target_key: str, target_obj_id: int,
                   flanker_type: str, split: str,
                   dataset_dir: Path, target_ids: list,
                   external_ids: list, rng: random.Random):
    if flanker_type == "same_class":
        pool = [p for p in get_pool(dataset_dir, split, target_obj_id)
                if p.stem != target_key]

    elif flanker_type == "different_class":
        others = [o for o in target_ids if o != target_obj_id]
        obj    = rng.choice(others)
        pool   = get_pool(dataset_dir, split, obj)

    elif flanker_type == "external":
        obj  = rng.choice(external_ids)
        pool = get_pool(dataset_dir, split, obj)

    else:
        return None

    return rng.choice(pool) if pool else None


def place_flanker_on_canvas(canvas: np.ndarray, flanker_img: np.ndarray,
                             center_x: int, center_y: int,
                             obj_size: int, bg_threshold: int = 20) -> np.ndarray:
    """Flanker nesnesini canvas üzerine yerleştirir. Siyah pikseller kopyalanmaz."""
    h, w  = canvas.shape[:2]
    half  = obj_size // 2
    fx0, fy0 = center_x - half, center_y - half
    fx1, fy1 = fx0 + obj_size, fy0 + obj_size

    cx0, cy0 = max(fx0, 0), max(fy0, 0)
    cx1, cy1 = min(fx1, w),  min(fy1, h)
    if cx0 >= cx1 or cy0 >= cy1:
        return canvas

    sx0 = cx0 - fx0
    sy0 = cy0 - fy0
    sx1 = sx0 + (cx1 - cx0)
    sy1 = sy0 + (cy1 - cy0)

    flanker_roi = flanker_img[sy0:sy1, sx0:sx1]
    canvas_roi  = canvas[cy0:cy1, cx0:cx1]
    mask        = np.any(flanker_roi > bg_threshold, axis=2)
    canvas_roi[mask] = flanker_roi[mask]
    canvas[cy0:cy1, cx0:cx1] = canvas_roi
    return canvas


def build_composite(target_path: Path, flanker_path: Path,
                    spacing_px: int, canvas_size: int,
                    object_size: int, bg_color: int) -> np.ndarray | None:
    target_img  = cv2.imread(str(target_path))
    flanker_img = cv2.imread(str(flanker_path))

    if target_img is None or flanker_img is None:
        return None

    canvas  = np.full((canvas_size, canvas_size, 3), bg_color, dtype=np.uint8)
    center  = canvas_size // 2
    half    = object_size // 2
    offset  = (canvas_size - object_size) // 2

    # Hedef — merkeze
    target_obj = target_img[offset:offset+object_size, offset:offset+object_size]
    canvas[center-half:center+half, center-half:center+half] = target_obj

    # Flanker nesne bölgesi
    flanker_obj = flanker_img[offset:offset+object_size, offset:offset+object_size]

    # Sol ve sağ — aynı flanker
    canvas = place_flanker_on_canvas(canvas, flanker_obj,
                                      center - spacing_px, center, object_size)
    canvas = place_flanker_on_canvas(canvas, flanker_obj,
                                      center + spacing_px, center, object_size)
    return canvas


def main(config_path: str):
    cfg          = load_config(config_path)
    dataset_dir  = Path(cfg["data"]["output_dir"])
    comp_dir     = dataset_dir / "composites"
    canvas_size  = cfg["image"]["canvas_size"]
    object_size  = cfg["image"]["object_size"]
    bg_color     = cfg["image"]["background_color"]
    target_ids   = cfg["data"]["target_obj_ids"]
    external_ids = cfg["data"]["external_obj_ids"]
    flanker_types = cfg["flanker"]["types"]
    spacings_deg  = cfg["spacing"]["degrees"]
    pix_per_deg   = cfg["spacing"]["pixels_per_degree"]
    samples       = cfg["data"]["samples_per_condition"]
    seed          = cfg["data"]["split"]["random_seed"]

    rng = random.Random(seed)

    if comp_dir.exists():
        shutil.rmtree(comp_dir)

    print("Leakage kontrolleri...")
    for split in ["val", "test"]:
        validate_no_leakage(dataset_dir, split)

    for split in ["val", "test"]:
        print(f"\n=== {split.upper()} ===")
        manifest = []

        for target_obj_id in target_ids:
            target_pool = get_pool(dataset_dir, split, target_obj_id)
            chosen      = rng.sample(target_pool, min(samples, len(target_pool)))

            for flanker_type in flanker_types:
                for target_path in chosen:
                    target_key = target_path.stem

                    # Flanker bir kez seç — tüm spacing'lerde sabit
                    flanker_path = select_flanker(
                        target_key, target_obj_id, flanker_type,
                        split, dataset_dir, target_ids, external_ids, rng
                    )
                    if flanker_path is None:
                        continue

                    for deg in spacings_deg:
                        spacing_px = int(deg * pix_per_deg)
                        out_dir    = (comp_dir / split / f"obj{target_obj_id}"
                                      / flanker_type / f"{deg}deg")
                        out_dir.mkdir(parents=True, exist_ok=True)

                        comp = build_composite(
                            target_path, flanker_path,
                            spacing_px, canvas_size, object_size, bg_color
                        )
                        if comp is None:
                            continue

                        out_path = out_dir / f"{target_key}.png"
                        cv2.imwrite(str(out_path), comp)
                        manifest.append({
                            "path":         str(out_path),
                            "target_obj":   target_obj_id,
                            "flanker_type": flanker_type,
                            "spacing_deg":  deg,
                            "key":          target_key,
                            "flanker_path": str(flanker_path),
                        })

            n = sum(1 for m in manifest if m["target_obj"] == target_obj_id)
            print(f"  obj{target_obj_id}: {n} kompozit")

        out_path = comp_dir / f"{split}_composite_manifest.json"
        with open(out_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"  Toplam: {len(manifest)} kompozit")

    print("\nTamamlandı.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
