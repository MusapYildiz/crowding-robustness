"""
data/coco/build_composites.py

COCO isolation gorsellerinden crowding kompozitleri uretir.

- Sol ve sag flanker: ayni instance (same_flanker_both_sides=true)
- Flanker her spacing seviyesinde sabit kalir
- Leakage kontrolu: flanker pool train setinden secilmez

Kullanim:
    python data/coco/build_composites.py --config configs/config_coco.yaml
"""

import json, random, argparse, shutil
import cv2
import numpy as np
from pathlib import Path
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_pool(dataset_dir: Path, split: str, category: str) -> list:
    d = dataset_dir / split / category
    return sorted(d.glob("*.png")) if d.exists() else []


def validate_no_leakage(dataset_dir: Path, split: str):
    train_keys = set(json.load(
        open(dataset_dir / "train_manifest.json")).keys())
    split_keys = set(json.load(
        open(dataset_dir / f"{split}_manifest.json")).keys())
    overlap = train_keys & split_keys
    if overlap:
        raise RuntimeError(
            f"[LEAKAGE] train n {split} = {len(overlap)} "
            f"-- ornek: {list(overlap)[:5]}"
        )
    print(f"  [OK] train n {split} = 0 ({len(split_keys)} instance)")


def select_flanker(target_ann_id: str, target_cat: str,
                   flanker_type: str, split: str,
                   dataset_dir: Path, categories: list,
                   external_categories: list,
                   rng: random.Random):
    if flanker_type == "same_class":
        pool = [p for p in get_pool(dataset_dir, split, target_cat)
                if p.stem != target_ann_id]

    elif flanker_type == "different_class":
        others = [c for c in categories if c != target_cat]
        cat    = rng.choice(others)
        pool   = get_pool(dataset_dir, split, cat)

    elif flanker_type == "external":
        cat  = rng.choice(external_categories)
        pool = get_pool(dataset_dir, split, cat)

    else:
        return None

    return rng.choice(pool) if pool else None


def place_flanker_on_canvas(canvas: np.ndarray,
                             flanker_obj: np.ndarray,
                             center_x: int, center_y: int,
                             obj_size: int,
                             bg_color: int) -> np.ndarray:
    h, w  = canvas.shape[:2]
    half  = obj_size // 2
    fx0   = center_x - half
    fy0   = center_y - half
    fx1   = fx0 + obj_size
    fy1   = fy0 + obj_size

    cx0, cy0 = max(fx0, 0), max(fy0, 0)
    cx1, cy1 = min(fx1, w),  min(fy1, h)
    if cx0 >= cx1 or cy0 >= cy1:
        return canvas

    sx0 = cx0 - fx0
    sy0 = cy0 - fy0
    sx1 = sx0 + (cx1 - cx0)
    sy1 = sy0 + (cy1 - cy0)

    flanker_roi = flanker_obj[sy0:sy1, sx0:sx1]
    canvas_roi  = canvas[cy0:cy1, cx0:cx1]

    # Sadece background olmayan pikselleri kopyala
    mask = np.any(flanker_roi != bg_color, axis=2)
    canvas_roi[mask] = flanker_roi[mask]
    canvas[cy0:cy1, cx0:cx1] = canvas_roi
    return canvas


def build_composite(target_path: Path, flanker_path: Path,
                    spacing_px: int, canvas_size: int,
                    object_size: int, bg_color: int):
    target_img  = cv2.imread(str(target_path))
    flanker_img = cv2.imread(str(flanker_path))

    if target_img is None or flanker_img is None:
        return None

    canvas  = np.full((canvas_size, canvas_size, 3), bg_color, dtype=np.uint8)
    center  = canvas_size // 2
    half    = object_size // 2
    offset  = (canvas_size - object_size) // 2

    # Hedef merkeze
    target_obj = target_img[offset:offset+object_size,
                             offset:offset+object_size]
    canvas[center-half:center+half, center-half:center+half] = target_obj

    # Flanker nesne bolgesini al
    flanker_obj = flanker_img[offset:offset+object_size,
                               offset:offset+object_size]

    # Sol ve sag -- ayni flanker
    canvas = place_flanker_on_canvas(canvas, flanker_obj,
                                      center - spacing_px, center,
                                      object_size, bg_color)
    canvas = place_flanker_on_canvas(canvas, flanker_obj,
                                      center + spacing_px, center,
                                      object_size, bg_color)
    return canvas


def main(config_path: str):
    cfg          = load_config(config_path)
    dataset_dir  = Path(cfg["data"]["output_dir"])
    comp_dir     = dataset_dir / "composites"
    canvas_size  = cfg["image"]["canvas_size"]
    object_size  = cfg["image"]["object_size"]
    bg_color     = cfg["image"]["background_color"]
    categories   = cfg["data"]["categories"]
    ext_cats     = cfg["data"]["external_flanker_categories"]
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

        for target_cat in categories:
            target_pool = get_pool(dataset_dir, split, target_cat)
            chosen      = rng.sample(target_pool,
                                     min(samples, len(target_pool)))

            for flanker_type in flanker_types:
                for target_path in chosen:
                    target_ann_id = target_path.stem

                    # Flanker bir kez sec -- tum spacing'lerde sabit
                    flanker_path = select_flanker(
                        target_ann_id, target_cat, flanker_type,
                        split, dataset_dir, categories, ext_cats, rng
                    )
                    if flanker_path is None:
                        continue

                    for deg in spacings_deg:
                        spacing_px = int(deg * pix_per_deg)
                        out_dir    = (comp_dir / split / target_cat
                                      / flanker_type / f"{deg}deg")
                        out_dir.mkdir(parents=True, exist_ok=True)

                        comp = build_composite(
                            target_path, flanker_path,
                            spacing_px, canvas_size, object_size, bg_color
                        )
                        if comp is None:
                            continue

                        out_path = out_dir / f"{target_ann_id}.png"
                        cv2.imwrite(str(out_path), comp)
                        manifest.append({
                            "path":         str(out_path),
                            "target_cat":   target_cat,
                            "flanker_type": flanker_type,
                            "spacing_deg":  deg,
                            "ann_id":       target_ann_id,
                            "flanker_path": str(flanker_path),
                        })

            n = sum(1 for m in manifest
                    if m["target_cat"] == target_cat)
            print(f"  {target_cat}: {n} kompozit")

        out_path = comp_dir / f"{split}_composite_manifest.json"
        with open(out_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"  Toplam: {len(manifest)} kompozit")

    print("\nTamamlandi.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_coco.yaml")
    args = parser.parse_args()
    main(args.config)
