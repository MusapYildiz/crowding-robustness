"""
data/openimages/build_composites.py

Isolation gorsellerinden crowding kompozitleri uretir.

Flanker tipleri:
  - congruent   : hedefle ayni sinif, farkli instance
  - incongruent : hedeften farkli sinif (veri setinden)
  - external    : veri setinin tamamen disinda (harici nesneler)

Her hedef icin flanker bir kez secilir,
tum spacing seviyelerinde ayni flanker kullanilir.
Sol ve sag flanker ayni instance.

Kullanim:
    python data/openimages/build_composites.py --config configs/config_openimages.yaml
"""

import json, random, argparse, shutil
import cv2
import numpy as np
from pathlib import Path
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_pool(dataset_dir: Path, split: str, label: str) -> list:
    d = dataset_dir / split / label
    return sorted(d.glob("*.png")) if d.exists() else []


def validate_no_leakage(dataset_dir: Path, split: str):
    train_keys = set(json.load(
        open(dataset_dir / "train_manifest.json")).keys())
    split_keys = set(json.load(
        open(dataset_dir / f"{split}_manifest.json")).keys())
    overlap = train_keys & split_keys
    if overlap:
        raise RuntimeError(
            f"[LEAKAGE] train n {split} = {len(overlap)}"
        )
    print(f"  [OK] train n {split} = 0 ({len(split_keys)} instance)")


def get_external_pool(dataset_dir: Path, split: str,
                       target_classes: list) -> list:
    """
    External flanker icin: veri setindeki tum siniflar disinda
    kalan gorseller. Yoksa incongruent ile ayni pool kullanilir.
    Burada basit yaklasim: en az gorulen siniflardan sec.
    """
    all_imgs = []
    for label_dir in sorted((dataset_dir / split).iterdir()):
        if label_dir.name not in target_classes:
            all_imgs.extend(sorted(label_dir.glob("*.png")))
    return all_imgs


def select_flanker(target_ann_id: str, target_label: str,
                   flanker_type: str, split: str,
                   dataset_dir: Path, target_classes: list,
                   rng: random.Random):
    if flanker_type == "congruent":
        pool = [p for p in get_pool(dataset_dir, split, target_label)
                if p.stem != target_ann_id]

    elif flanker_type == "incongruent":
        others = [c for c in target_classes if c != target_label]
        label  = rng.choice(others)
        pool   = get_pool(dataset_dir, split, label)

    elif flanker_type == "external":
        pool = get_external_pool(dataset_dir, split, target_classes)
        if not pool:
            # Fallback: incongruent
            others = [c for c in target_classes if c != target_label]
            label  = rng.choice(others)
            pool   = get_pool(dataset_dir, split, label)

    else:
        return None

    return rng.choice(pool) if pool else None


def place_flanker_on_canvas(canvas: np.ndarray,
                             flanker_obj: np.ndarray,
                             center_x: int, center_y: int,
                             long_edge: int,
                             bg_color: int) -> np.ndarray:
    """
    Flanker nesnesini canvas uzerine yerlestirir.
    Siyah (bg_color) olan pikseller kopyalanmaz.
    Canvas disina taşan kisimlar kirpilir (partial flanker).
    """
    h, w   = canvas.shape[:2]
    half   = long_edge // 2
    fx0    = center_x - half
    fy0    = center_y - half
    fx1    = fx0 + long_edge
    fy1    = fy0 + long_edge

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
    mask = np.any(flanker_roi > bg_color + 10, axis=2)
    canvas_roi[mask] = flanker_roi[mask]
    canvas[cy0:cy1, cx0:cx1] = canvas_roi
    return canvas


def build_composite(target_path: Path, flanker_path: Path,
                    spacing_px: int, canvas_size: int,
                    long_edge: int, bg_color: int):
    target_img  = cv2.imread(str(target_path))
    flanker_img = cv2.imread(str(flanker_path))

    if target_img is None or flanker_img is None:
        return None

    canvas = np.full((canvas_size, canvas_size, 3),
                      bg_color, dtype=np.uint8)
    center = canvas_size // 2
    half   = long_edge // 2
    offset = (canvas_size - long_edge) // 2

    # Hedef merkeze
    target_obj  = target_img[offset:offset+long_edge,
                               offset:offset+long_edge]
    canvas[center-half:center+half,
           center-half:center+half] = target_obj

    # Flanker nesne bolgesini al
    flanker_obj = flanker_img[offset:offset+long_edge,
                               offset:offset+long_edge]

    # Sol ve sag -- ayni flanker
    canvas = place_flanker_on_canvas(
        canvas, flanker_obj,
        center - spacing_px, center, long_edge, bg_color
    )
    canvas = place_flanker_on_canvas(
        canvas, flanker_obj,
        center + spacing_px, center, long_edge, bg_color
    )
    return canvas


def main(config_path: str):
    cfg           = load_config(config_path)
    dataset_dir   = Path(cfg["data"]["output_dir"])
    comp_dir      = dataset_dir / "composites"
    canvas_size   = cfg["image"]["canvas_size"]
    long_edge     = cfg["image"]["object_long_edge"]
    bg_color      = cfg["image"]["background_color"]
    target_classes = cfg["data"]["target_classes"]
    flanker_types  = cfg["flanker"]["types"]
    spacings_deg   = cfg["spacing"]["degrees"]
    pix_per_deg    = cfg["spacing"]["pixels_per_degree"]
    samples        = cfg["data"]["samples_per_condition"]
    seed           = cfg["data"]["split"]["random_seed"]

    rng = random.Random(seed)

    if comp_dir.exists():
        shutil.rmtree(comp_dir)

    print("Leakage kontrolleri...")
    for split in ["val", "test"]:
        validate_no_leakage(dataset_dir, split)

    for split in ["val", "test"]:
        print(f"\n=== {split.upper()} ===")
        manifest = []

        for target_label in target_classes:
            target_pool = get_pool(dataset_dir, split, target_label)
            chosen      = rng.sample(target_pool,
                                     min(samples, len(target_pool)))

            for flanker_type in flanker_types:
                for target_path in chosen:
                    target_ann_id = target_path.stem

                    # Flanker bir kez sec -- tum spacing'lerde sabit
                    flanker_path = select_flanker(
                        target_ann_id, target_label,
                        flanker_type, split,
                        dataset_dir, target_classes, rng
                    )
                    if flanker_path is None:
                        continue

                    for deg in spacings_deg:
                        spacing_px = int(deg * pix_per_deg)
                        out_dir    = (comp_dir / split / target_label
                                      / flanker_type / f"{deg}deg")
                        out_dir.mkdir(parents=True, exist_ok=True)

                        comp = build_composite(
                            target_path, flanker_path,
                            spacing_px, canvas_size,
                            long_edge, bg_color
                        )
                        if comp is None:
                            continue

                        out_path = out_dir / f"{target_ann_id}.png"
                        cv2.imwrite(str(out_path), comp)
                        manifest.append({
                            "path":         str(out_path),
                            "target_label": target_label,
                            "flanker_type": flanker_type,
                            "spacing_deg":  deg,
                            "ann_id":       target_ann_id,
                            "flanker_path": str(flanker_path),
                        })

            n = sum(1 for m in manifest
                    if m["target_label"] == target_label)
            print(f"  {target_label}: {n} kompozit")

        out_path = comp_dir / f"{split}_composite_manifest.json"
        with open(out_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"  Toplam: {len(manifest)} kompozit")

    print("\nTamamlandi.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_openimages.yaml")
    args = parser.parse_args()
    main(args.config)
