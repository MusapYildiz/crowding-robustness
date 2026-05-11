"""
prepare_dataset.py

COCO 2017'den hedef kategorilerin instance'larını okur,
instance bazlı strict split uygular (train/val/test),
her instance'ı segmentasyon maskesiyle arka plandan temizler,
uniform background üzerine sabit boyutta kaydeder.

Kullanım:
    python data/prepare_dataset.py --config configs/config.yaml
"""

import os
import json
import random
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import cv2
import yaml
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils
from tqdm import tqdm


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_category_ids(coco: COCO, category_names: list) -> dict:
    """Kategori adlarından COCO category ID'lerini döndürür."""
    name_to_id = {}
    for name in category_names:
        # config'deki alt çizgileri boşluğa çevir (fire_hydrant -> fire hydrant)
        coco_name = name.replace("_", " ")
        cats = coco.getCatIds(catNms=[coco_name])
        if not cats:
            print(f"  [WARN] Kategori bulunamadı: '{coco_name}', atlanıyor.")
            continue
        name_to_id[name] = cats[0]
        print(f"  '{coco_name}' -> category_id={cats[0]}")
    return name_to_id


def collect_instances(coco: COCO, category_id: int, min_bbox_area: int) -> list:
    """
    Bir kategori için tüm geçerli annotation'ları toplar.
    Minimum alan filtresi uygular.
    Her instance: {ann_id, image_id, bbox, segmentation, area}
    """
    ann_ids = coco.getAnnIds(catIds=[category_id], iscrowd=False)
    anns = coco.loadAnns(ann_ids)

    instances = []
    for ann in anns:
        x, y, w, h = ann["bbox"]
        area = w * h
        if area < min_bbox_area:
            continue
        if w < 1 or h < 1:
            continue
        instances.append({
            "ann_id":       ann["id"],
            "image_id":     ann["image_id"],
            "bbox":         ann["bbox"],       # [x, y, w, h]
            "segmentation": ann["segmentation"],
            "area":         area,
            "category_id":  category_id,
        })
    return instances


def split_instances(instances: list, train_ratio: float, val_ratio: float,
                    test_ratio: float, seed: int) -> dict:
    """
    Instance bazlı strict split.
    Aynı instance train, val ve test'te aynı anda yer alamaz.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Oranların toplamı 1 olmalı."

    rng = random.Random(seed)
    shuffled = instances[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    return {
        "train": shuffled[:n_train],
        "val":   shuffled[n_train:n_train + n_val],
        "test":  shuffled[n_train + n_val:],
    }


def extract_masked_object(image: np.ndarray, ann: dict, coco: COCO):
    """
    Segmentasyon maskesi kullanarak nesneyi arka plandan temizler.
    Döndürür: (crop_bgr, binary_mask) tuple veya None.
    """
    h_img, w_img = image.shape[:2]

    # Maske oluştur
    rle         = coco.annToRLE(ann)
    binary_mask = mask_utils.decode(rle).astype(np.uint8)

    # Bounding box — float -> int
    x, y, w, h = ann["bbox"]
    x, y       = int(x), int(y)
    w, h       = max(int(w), 1), max(int(h), 1)
    x2, y2     = min(x + w, w_img), min(y + h, h_img)

    if x2 <= x or y2 <= y:
        return None

    crop_rgb  = image[y:y2, x:x2]
    crop_mask = binary_mask[y:y2, x:x2]

    if crop_rgb.size == 0 or crop_mask.sum() == 0:
        return None

    return crop_rgb, crop_mask


def place_on_canvas(obj_tuple, object_size: int,
                    canvas_size: int, bg_color: int) -> np.ndarray:
    """
    (crop_bgr, crop_mask) tuple alinir:
      - object_size x object_size'a resize edilir
      - canvas_size x canvas_size uniform background'un merkezine yerlestirilir
      - Sadece maske pikselleri canvas'a kopyalanir
    Döndürür: (canvas_size, canvas_size, 3) uint8 BGR görsel.
    """
    crop_rgb, crop_mask = obj_tuple

    resized_obj  = cv2.resize(crop_rgb,  (object_size, object_size),
                              interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(crop_mask, (object_size, object_size),
                              interpolation=cv2.INTER_NEAREST)

    canvas = np.full((canvas_size, canvas_size, 3), bg_color, dtype=np.uint8)
    offset = (canvas_size - object_size) // 2

    roi = canvas[offset:offset+object_size, offset:offset+object_size]
    roi[resized_mask > 0] = resized_obj[resized_mask > 0]
    canvas[offset:offset+object_size, offset:offset+object_size] = roi

    return canvas


def process_category(coco: COCO, category_name: str, category_id: int,
                     split_data: dict, output_dir: Path, cfg: dict):
    """
    Bir kategorinin tüm split'lerindeki instance'larını işler ve kaydeder.
    """
    img_cfg = cfg["image"]
    canvas_size = img_cfg["canvas_size"]
    object_size = img_cfg["object_size"]
    bg_color    = img_cfg["background_color"]

    for split_name, instances in split_data.items():
        split_dir = output_dir / split_name / category_name
        split_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        for inst in tqdm(instances, desc=f"  {split_name}/{category_name}", leave=False):
            # Görseli yükle
            img_info = coco.loadImgs(inst["image_id"])[0]
            img_path = Path(cfg["data"]["coco_dir"]) / "images" / "train2017" / img_info["file_name"]

            if not img_path.exists():
                continue

            image = cv2.imread(str(img_path))
            if image is None:
                continue

            # Maske ile kes
            ann = coco.loadAnns(inst["ann_id"])[0]
            rgba = extract_masked_object(image, ann, coco)
            if rgba is None:
                continue

            # Canvas'a yerleştir
            canvas = place_on_canvas(rgba, object_size, canvas_size, bg_color)

            # Kaydet: {ann_id}.png
            out_path = split_dir / f"{inst['ann_id']}.png"
            cv2.imwrite(str(out_path), canvas)
            saved += 1

        print(f"    {split_name}/{category_name}: {saved} görsel kaydedildi")


def save_split_manifest(split_data_all: dict, output_dir: Path):
    """
    Her split için ann_id → kategori eşleşmesini JSON olarak kaydeder.
    Sonraki adımlarda (composite oluşturma) kullanılır.
    """
    for split_name in ["train", "val", "test"]:
        manifest = {}
        for category_name, split_data in split_data_all.items():
            for inst in split_data.get(split_name, []):
                manifest[str(inst["ann_id"])] = {
                    "category": category_name,
                    "image_id": inst["image_id"],
                    "bbox":     inst["bbox"],
                    "area":     inst["area"],
                }
        out_path = output_dir / f"{split_name}_manifest.json"
        with open(out_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"  Manifest kaydedildi: {out_path} ({len(manifest)} instance)")


def main(config_path: str):
    cfg = load_config(config_path)

    coco_ann_path = (
        Path(cfg["data"]["coco_dir"]) / "annotations" / cfg["data"]["annotation_file"]
    )
    output_dir = Path(cfg["data"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"COCO annotation yükleniyor: {coco_ann_path}")
    coco = COCO(str(coco_ann_path))

    # Tüm hedef kategoriler + external flanker kategoriler
    all_category_names = (
        cfg["data"]["categories"] + cfg["data"]["external_flanker_categories"]
    )

    print("\nKategori ID'leri alınıyor...")
    name_to_id = get_category_ids(coco, all_category_names)

    split_cfg = cfg["data"]["split"]
    split_data_all = {}  # {category_name: {train: [...], val: [...], test: [...]}}

    print("\nInstance'lar toplanıyor ve split yapılıyor...")
    for cat_name, cat_id in name_to_id.items():
        instances = collect_instances(
            coco, cat_id, cfg["image"]["min_bbox_area"]
        )
        print(f"  {cat_name}: {len(instances)} geçerli instance")

        splits = split_instances(
            instances,
            train_ratio=split_cfg["train_ratio"],
            val_ratio=split_cfg["val_ratio"],
            test_ratio=split_cfg["test_ratio"],
            seed=split_cfg["random_seed"],
        )
        print(f"    train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
        split_data_all[cat_name] = splits

    print("\nGörseller işleniyor...")
    for cat_name, cat_id in name_to_id.items():
        print(f"\n[{cat_name}]")
        process_category(
            coco, cat_name, cat_id,
            split_data_all[cat_name],
            output_dir, cfg,
        )

    print("\nManifest'ler kaydediliyor...")
    save_split_manifest(split_data_all, output_dir)

    print("\nTamamlandı.")
    print(f"Çıktı dizini: {output_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COCO'dan crowding dataset'i hazırla")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
