"""
build_composites.py

Hazırlanan nesne görsellerinden crowding test kompozitleri oluşturur.

Her test görseli:
  - Merkeze hedef nesne (target)
  - Soluna ve sağına flanker (spacing: 1°, 2°, 4°, 8°)

Flanker tipleri:
  1. same_class    : hedef ile aynı kategoriden, farklı instance
  2. different_class: veri setindeki farklı kategori
  3. external      : external_flanker_categories'den

Kullanım:
    python data/build_composites.py --config configs/config.yaml
"""

import os
import json
import random
import argparse
import numpy as np
from pathlib import Path
from itertools import combinations

import cv2
import yaml
from tqdm import tqdm


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_manifest(dataset_dir: Path, split: str) -> dict:
    path = dataset_dir / f"{split}_manifest.json"
    with open(path, "r") as f:
        return json.load(f)


def get_image_pool(dataset_dir: Path, split: str, category: str) -> list:
    """Bir kategorinin split'indeki tüm görsel yollarını döndürür."""
    cat_dir = dataset_dir / split / category
    if not cat_dir.exists():
        return []
    return sorted(cat_dir.glob("*.png"))


def build_image_pools(dataset_dir: Path, split: str, categories: list,
                      external_categories: list) -> dict:
    """
    Tüm kategoriler için görsel pool'larını hazırlar.
    {category_name: [Path, Path, ...]}
    """
    pools = {}
    all_cats = categories + external_categories
    for cat in all_cats:
        pools[cat] = get_image_pool(dataset_dir, split, cat)
    return pools


def degrees_to_pixels(degrees: float, pixels_per_degree: float) -> int:
    return int(round(degrees * pixels_per_degree))


def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    return img


def place_flanker(canvas: np.ndarray, flanker_img: np.ndarray,
                  center_x: int, center_y: int, object_size: int) -> np.ndarray:
    """
    Flanker'ı canvas üzerine yerleştirir.
    center_x, center_y: flanker'ın merkez koordinatları.
    Canvas dışına taşan kısımlar kırpılır (partial flanker).
    """
    canvas_h, canvas_w = canvas.shape[:2]
    half = object_size // 2

    # Flanker koordinatları (canvas koordinatlarında)
    fx0 = center_x - half
    fy0 = center_y - half
    fx1 = fx0 + object_size
    fy1 = fy0 + object_size

    # Flanker crop sınırları (canvas dışına taşarsa kırp)
    cx0 = max(fx0, 0)
    cy0 = max(fy0, 0)
    cx1 = min(fx1, canvas_w)
    cy1 = min(fy1, canvas_h)

    if cx0 >= cx1 or cy0 >= cy1:
        return canvas  # tamamen dışarıda, bir şey yapma

    # Flanker üzerindeki karşılık gelen bölge
    src_x0 = cx0 - fx0
    src_y0 = cy0 - fy0
    src_x1 = src_x0 + (cx1 - cx0)
    src_y1 = src_y0 + (cy1 - cy0)

    canvas[cy0:cy1, cx0:cx1] = flanker_img[src_y0:src_y1, src_x0:src_x1]
    return canvas


def build_composite(target_path: Path, left_flanker_path: Path,
                    right_flanker_path: Path, spacing_px: int,
                    canvas_size: int, object_size: int,
                    bg_color: int) -> np.ndarray:
    """
    Hedef + sol flanker + sağ flanker kompozit görselini oluşturur.
    """
    target_img  = load_image(target_path)
    left_img    = load_image(left_flanker_path)
    right_img   = load_image(right_flanker_path)

    canvas = np.full((canvas_size, canvas_size, 3), bg_color, dtype=np.uint8)

    center = canvas_size // 2

    # Hedef: merkeze
    half = object_size // 2
    t0, t1 = center - half, center + half
    canvas[t0:t1, t0:t1] = target_img[t0:t1, t0:t1]

    # Sol flanker merkezi: target_center_x - spacing_px
    left_cx  = center - spacing_px
    right_cx = center + spacing_px

    canvas = place_flanker(canvas, left_img,  left_cx,  center, object_size)
    canvas = place_flanker(canvas, right_img, right_cx, center, object_size)

    return canvas


def select_flanker(target_ann_id: str, target_category: str,
                   flanker_type: str, pools: dict,
                   categories: list, external_categories: list,
                   rng: random.Random) -> Path | None:
    """
    Flanker tipi ve kurallara göre bir flanker görsel seçer.
    - same_class: aynı kategori, farklı instance (farklı dosya adı)
    - different_class: farklı kategori (veri setinden)
    - external: external_flanker_categories'den
    """
    if flanker_type == "same_class":
        pool = [p for p in pools[target_category]
                if p.stem != target_ann_id]
        if not pool:
            return None
        return rng.choice(pool)

    elif flanker_type == "different_class":
        other_cats = [c for c in categories if c != target_category]
        if not other_cats:
            return None
        chosen_cat = rng.choice(other_cats)
        pool = pools.get(chosen_cat, [])
        if not pool:
            return None
        return rng.choice(pool)

    elif flanker_type == "external":
        chosen_cat = rng.choice(external_categories)
        pool = pools.get(chosen_cat, [])
        if not pool:
            return None
        return rng.choice(pool)

    return None


def validate_no_leakage(dataset_dir: Path, split: str) -> None:
    """
    Flanker olarak kullanılacak pool'un (test/val) ile
    train manifest'inin hiç kesişmediğini doğrular.

    Her instance dosya adı = ann_id.png
    Train manifest'indeki ann_id'ler ile split pool'undaki ann_id'ler
    aynı anda bulunamaz.

    Hata varsa RuntimeError fırlatır, program durur.
    """
    train_manifest_path = dataset_dir / "train_manifest.json"
    split_manifest_path = dataset_dir / f"{split}_manifest.json"

    if not train_manifest_path.exists():
        raise FileNotFoundError(f"Train manifest bulunamadı: {train_manifest_path}")
    if not split_manifest_path.exists():
        raise FileNotFoundError(f"{split} manifest bulunamadı: {split_manifest_path}")

    with open(train_manifest_path) as f:
        train_ids = set(json.load(f).keys())  # ann_id string set

    with open(split_manifest_path) as f:
        split_ids = set(json.load(f).keys())

    overlap = train_ids & split_ids

    if overlap:
        raise RuntimeError(
            f"\n[LEAKAGE DETECTED] {len(overlap)} instance hem train hem {split} "
            f"split'inde bulunuyor!\n"
            f"Örnek çakışan ann_id'ler: {list(overlap)[:10]}\n"
            f"Pipeline durduruluyor — lütfen split mantığını kontrol edin."
        )

    print(f"  [OK] Leakage kontrolü geçti: train ∩ {split} = ∅ "
          f"(train={len(train_ids)}, {split}={len(split_ids)})")


def build_split_composites(split: str, dataset_dir: Path, output_dir: Path,
                           cfg: dict, rng: random.Random):
    """
    Bir split (test/val) için tüm kompozit görsellerini oluşturur.
    Sadece test ve val için composite oluşturulur —
    train seti zaten tek nesneli (isolation) görsellerden oluşur.
    """
    categories          = cfg["data"]["categories"]
    external_categories = cfg["data"]["external_flanker_categories"]
    flanker_types       = cfg["flanker"]["types"]
    spacing_degrees     = cfg["spacing"]["degrees"]
    pixels_per_degree   = cfg["spacing"]["pixels_per_degree"]
    samples             = cfg["data"]["samples_per_condition"]
    canvas_size         = cfg["image"]["canvas_size"]
    object_size         = cfg["image"]["object_size"]
    bg_color            = cfg["image"]["background_color"]

    pools = build_image_pools(dataset_dir, split, categories, external_categories)

    for target_cat in tqdm(categories, desc=f"[{split}] Kategoriler"):
        target_pool = pools[target_cat]
        if len(target_pool) < samples:
            print(f"  [WARN] {target_cat}: yeterli görsel yok "
                  f"({len(target_pool)} < {samples}), mevcut kadarıyla devam edilecek.")

        # Her koşul için samples kadar hedef seç
        chosen_targets = rng.choices(target_pool, k=min(samples, len(target_pool)))

        for flanker_type in flanker_types:
            for deg in spacing_degrees:
                spacing_px = degrees_to_pixels(deg, pixels_per_degree)

                # Çıktı dizini: output/split/target_cat/flanker_type/deg/
                cond_dir = (output_dir / split / target_cat /
                            flanker_type / f"{deg}deg")
                cond_dir.mkdir(parents=True, exist_ok=True)

                saved = 0
                for target_path in chosen_targets:
                    target_ann_id = target_path.stem

                    left_path = select_flanker(
                        target_ann_id, target_cat, flanker_type,
                        pools, categories, external_categories, rng
                    )
                    right_path = select_flanker(
                        target_ann_id, target_cat, flanker_type,
                        pools, categories, external_categories, rng
                    )

                    if left_path is None or right_path is None:
                        continue

                    composite = build_composite(
                        target_path, left_path, right_path,
                        spacing_px, canvas_size, object_size, bg_color
                    )

                    out_path = cond_dir / f"{target_ann_id}.png"
                    cv2.imwrite(str(out_path), composite)
                    saved += 1

                print(f"  {split}/{target_cat}/{flanker_type}/{deg}°: {saved} görsel")


def build_composite_manifest(composite_dir: Path, output_dir: Path, split: str):
    """
    Oluşturulan kompozit görsellerinin metadata'sını JSON olarak kaydeder.
    """
    manifest = []
    for img_path in sorted(composite_dir.rglob("*.png")):
        parts = img_path.relative_to(composite_dir / split).parts
        if len(parts) < 4:
            continue
        target_cat, flanker_type, deg_str, filename = parts
        manifest.append({
            "path":          str(img_path),
            "target_class":  target_cat,
            "flanker_type":  flanker_type,
            "spacing_deg":   float(deg_str.replace("deg", "")),
            "ann_id":        Path(filename).stem,
        })

    out_path = output_dir / f"{split}_composite_manifest.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Composite manifest: {out_path} ({len(manifest)} görsel)")


def main(config_path: str):
    cfg = load_config(config_path)
    dataset_dir = Path(cfg["data"]["output_dir"])
    composite_dir = Path(cfg["data"]["output_dir"]) / "composites"
    composite_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(cfg["data"]["split"]["random_seed"])

    # --- Leakage kontrolü: flanker pool'u train ile kesişmemeli ---
    print("\nLeakage kontrolleri yapılıyor...")
    for split in ["val", "test"]:
        validate_no_leakage(dataset_dir, split)

    for split in ["val", "test"]:
        print(f"\n=== {split.upper()} kompozitleri oluşturuluyor ===")
        build_split_composites(split, dataset_dir, composite_dir, cfg, rng)
        build_composite_manifest(composite_dir, composite_dir, split)

    print("\nTamamlandı.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crowding kompozit görseller oluştur")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
