"""
data/coco/prepare_dataset.py

COCO 2017'den hedef kategorilerin instance'larini okur,
instance bazli strict split uygular (train/val/test),
her instance'i segmentasyon maskesiyle arkaPlandan temizler,
aspect ratio koruyarak uniform gri background uzerine kaydeder.

Kullanim:
    python data/coco/prepare_dataset.py --config configs/config_coco.yaml
"""

import os, json, random, argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import cv2
import yaml
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils
from tqdm import tqdm


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_category_ids(coco: COCO, category_names: list) -> dict:
    name_to_id = {}
    for name in category_names:
        coco_name = name.replace("_", " ")
        cats = coco.getCatIds(catNms=[coco_name])
        if not cats:
            print(f"  [WARN] Kategori bulunamadi: '{coco_name}'")
            continue
        name_to_id[name] = cats[0]
        print(f"  '{coco_name}' -> category_id={cats[0]}")
    return name_to_id


def is_valid_instance(ann: dict, image: np.ndarray,
                      binary_mask: np.ndarray,
                      min_bbox_area: int,
                      min_mask_ratio: float,
                      min_color_std: float) -> bool:
    """Bozuk veya kalitesiz instance'lari filtreler."""
    x, y, w, h = ann["bbox"]
    x, y = int(x), int(y)
    w, h = max(int(w), 1), max(int(h), 1)

    h_img, w_img = image.shape[:2]
    x2 = min(x + w, w_img)
    y2 = min(y + h, h_img)

    if (x2 - x) * (y2 - y) < min_bbox_area:
        return False

    crop_mask = binary_mask[y:y2, x:x2]
    crop_rgb  = image[y:y2, x:x2]
    bbox_area = (x2 - x) * (y2 - y)

    # Maske doluluk orani
    if bbox_area == 0 or crop_mask.sum() / bbox_area < min_mask_ratio:
        return False

    # Renk cesitliligi
    masked_pixels = crop_rgb[crop_mask > 0]
    if len(masked_pixels) == 0 or masked_pixels.std() < min_color_std:
        return False

    return True


def collect_instances(coco: COCO, category_id: int,
                      cfg: dict, downloaded_img_ids: set) -> list:
    """Bir kategori icin gecerli annotation'lari toplar."""
    min_bbox_area = cfg["image"]["min_bbox_area"]
    min_mask_ratio = cfg["image"]["min_mask_ratio"]
    min_color_std  = cfg["image"]["min_color_std"]
    coco_dir       = Path(cfg["data"]["coco_dir"])

    ann_ids = coco.getAnnIds(catIds=[category_id], iscrowd=False)
    anns    = coco.loadAnns(ann_ids)

    instances = []
    for ann in anns:
        if ann["image_id"] not in downloaded_img_ids:
            continue

        x, y, w, h = ann["bbox"]
        if w * h < min_bbox_area:
            continue

        # Gorsel yukle ve kalite kontrolu yap
        img_info = coco.loadImgs(ann["image_id"])[0]
        img_path = coco_dir / "images" / "train2017" / img_info["file_name"]
        if not img_path.exists():
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        rle         = coco.annToRLE(ann)
        binary_mask = mask_utils.decode(rle).astype(np.uint8)

        if not is_valid_instance(ann, image, binary_mask,
                                  min_bbox_area, min_mask_ratio, min_color_std):
            continue

        instances.append({
            "ann_id":      ann["id"],
            "image_id":    ann["image_id"],
            "bbox":        ann["bbox"],
            "category_id": category_id,
        })

    return instances


def split_instances(instances: list, train_ratio: float,
                    val_ratio: float, seed: int) -> dict:
    rng = random.Random(seed)
    shuffled = instances[:]
    rng.shuffle(shuffled)
    n       = len(shuffled)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    return {
        "train": shuffled[:n_train],
        "val":   shuffled[n_train:n_train + n_val],
        "test":  shuffled[n_train + n_val:],
    }


def extract_and_place(image: np.ndarray, ann: dict, coco: COCO,
                      object_size: int, canvas_size: int,
                      bg_color: int) -> np.ndarray | None:
    """
    Segmentasyon maskesiyle nesneyi keser,
    aspect ratio koruyarak canvas'a yerlestirir.
    """
    h_img, w_img = image.shape[:2]

    rle         = coco.annToRLE(ann)
    binary_mask = mask_utils.decode(rle).astype(np.uint8)

    x, y, w, h = ann["bbox"]
    x, y = int(x), int(y)
    w, h = max(int(w), 1), max(int(h), 1)
    x2   = min(x + w, w_img)
    y2   = min(y + h, h_img)

    if x2 <= x or y2 <= y:
        return None

    crop_rgb  = image[y:y2, x:x2]
    crop_mask = binary_mask[y:y2, x:x2]

    if crop_rgb.size == 0 or crop_mask.sum() == 0:
        return None

    # Aspect ratio koruyarak resize
    ch, cw = crop_rgb.shape[:2]
    scale  = object_size / max(ch, cw)
    new_w  = max(int(cw * scale), 1)
    new_h  = max(int(ch * scale), 1)

    resized_obj  = cv2.resize(crop_rgb,  (new_w, new_h), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(crop_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # Canvas
    canvas   = np.full((canvas_size, canvas_size, 3), bg_color, dtype=np.uint8)
    offset_y = (canvas_size - new_h) // 2
    offset_x = (canvas_size - new_w) // 2

    roi = canvas[offset_y:offset_y+new_h, offset_x:offset_x+new_w]
    roi[resized_mask > 0] = resized_obj[resized_mask > 0]
    canvas[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = roi

    return canvas


def process_category(coco: COCO, category_name: str,
                     split_data: dict, output_dir: Path, cfg: dict):
    coco_dir    = Path(cfg["data"]["coco_dir"])
    canvas_size = cfg["image"]["canvas_size"]
    object_size = cfg["image"]["object_size"]
    bg_color    = cfg["image"]["background_color"]

    for split_name, instances in split_data.items():
        split_dir = output_dir / split_name / category_name
        split_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        for inst in tqdm(instances,
                         desc=f"  {split_name}/{category_name}",
                         leave=False):
            img_info = coco.loadImgs(inst["image_id"])[0]
            img_path = (coco_dir / "images" / "train2017" /
                        img_info["file_name"])
            if not img_path.exists():
                continue

            image = cv2.imread(str(img_path))
            if image is None:
                continue

            ann    = coco.loadAnns(inst["ann_id"])[0]
            canvas = extract_and_place(image, ann, coco,
                                        object_size, canvas_size, bg_color)
            if canvas is None:
                continue

            out_path = split_dir / f"{inst['ann_id']}.png"
            cv2.imwrite(str(out_path), canvas)
            saved += 1

        print(f"    {split_name}/{category_name}: {saved} gorsel")


def save_manifests(split_data_all: dict, output_dir: Path):
    for split_name in ["train", "val", "test"]:
        manifest = {}
        for cat_name, splits in split_data_all.items():
            for inst in splits.get(split_name, []):
                manifest[str(inst["ann_id"])] = {
                    "category": cat_name,
                    "image_id": inst["image_id"],
                    "bbox":     inst["bbox"],
                }
        out = output_dir / f"{split_name}_manifest.json"
        with open(out, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"  {split_name}_manifest: {len(manifest)} instance")


def main(config_path: str):
    cfg = load_config(config_path)

    ann_path   = (Path(cfg["data"]["coco_dir"]) / "annotations" /
                  cfg["data"]["annotation_file"])
    output_dir = Path(cfg["data"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"COCO annotation yukleniyor: {ann_path}")
    coco = COCO(str(ann_path))

    # Indirilen gorsellerin ID'leri
    img_dir = Path(cfg["data"]["coco_dir"]) / "images" / "train2017"
    downloaded_filenames = {p.name for p in img_dir.glob("*.jpg")}
    downloaded_img_ids   = {
        img["id"] for img in coco.dataset["images"]
        if img["file_name"] in downloaded_filenames
    }
    print(f"Indirilen gorsel: {len(downloaded_img_ids)}")

    all_categories = (cfg["data"]["categories"] +
                      cfg["data"]["external_flanker_categories"])

    print("\nKategori ID'leri aliniyor...")
    name_to_id = get_category_ids(coco, all_categories)

    split_cfg = cfg["data"]["split"]
    split_data_all = {}

    print("\nInstance'lar toplanip split yapiliyor...")
    for cat_name, cat_id in name_to_id.items():
        instances = collect_instances(coco, cat_id, cfg, downloaded_img_ids)
        print(f"  {cat_name}: {len(instances)} gecerli instance")
        splits = split_instances(
            instances,
            train_ratio=split_cfg["train_ratio"],
            val_ratio=split_cfg["val_ratio"],
            seed=split_cfg["random_seed"],
        )
        print(f"    train={len(splits['train'])} "
              f"val={len(splits['val'])} "
              f"test={len(splits['test'])}")
        split_data_all[cat_name] = splits

    print("\nGorseller isleniyor...")
    for cat_name in name_to_id:
        print(f"\n[{cat_name}]")
        process_category(coco, cat_name,
                         split_data_all[cat_name], output_dir, cfg)

    # Leakage kontrol
    print("\nLeakage kontrolleri...")
    train_ids = set()
    for splits in split_data_all.values():
        train_ids.update(str(i["ann_id"]) for i in splits["train"])

    for split_name in ["val", "test"]:
        split_ids = set()
        for splits in split_data_all.values():
            split_ids.update(str(i["ann_id"]) for i in splits[split_name])
        overlap = train_ids & split_ids
        if overlap:
            raise RuntimeError(f"LEAKAGE: train n {split_name} = {len(overlap)}")
        print(f"  [OK] train n {split_name} = 0")

    print("\nManifest'ler kaydediliyor...")
    save_manifests(split_data_all, output_dir)
    print("\nTamamlandi.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_coco.yaml")
    args = parser.parse_args()
    main(args.config)
