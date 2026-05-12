"""
data/openimages/download.py

Open Images v7'den hedef kategorilerin segmentasyonlu gorsellerini indirir.
Her gorsel icin:
  - Instance segmentasyon maskesi
  - Orijinal gorsel
  - Annotation metadata

Kullanim:
    python data/openimages/download.py --config configs/config_openimages.yaml

Gereksinimler:
    pip install fiftyone
"""

import os, json, argparse
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def download_openimages(cfg: dict):
    try:
        import fiftyone as fo
        import fiftyone.zoo as foz
    except ImportError:
        raise ImportError("fiftyone gerekli: pip install fiftyone")

    target_classes  = cfg["data"]["target_classes"]
    max_per_class   = cfg["data"]["max_samples_per_class"]
    output_dir      = Path(cfg["data"]["output_dir"])
    raw_dir         = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"Open Images v7 indiriliyor...")
    print(f"  Siniflar: {len(target_classes)}")
    print(f"  Sinif basi max: {max_per_class}")
    print(f"  Cikti: {raw_dir}")

    # fiftyone ile indir
    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split="train",
        label_types=["segmentations"],
        classes=target_classes,
        max_samples=max_per_class * len(target_classes),
        dataset_dir=str(raw_dir),
        dataset_name="crowding_openimages",
    )

    print(f"\nIndirilen toplam gorsel: {len(dataset)}")

    # Sinif bazli istatistik
    class_counts = {}
    for sample in dataset:
        if sample.ground_truth is None:
            continue
        for det in sample.ground_truth.detections:
            label = det.label
            if label in target_classes:
                class_counts[label] = class_counts.get(label, 0) + 1

    print("\nSinif bazli instance sayisi:")
    for cls in target_classes:
        count = class_counts.get(cls, 0)
        status = "OK" if count >= 50 else "AZ"
        print(f"  [{status}] {cls}: {count}")

    # Dataset bilgisini kaydet
    info = {
        "total_samples": len(dataset),
        "class_counts":  class_counts,
        "dataset_dir":   str(raw_dir),
    }
    with open(output_dir / "download_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nBilgi kaydedildi: {output_dir / 'download_info.json'}")
    return dataset


def main(config_path: str):
    cfg = load_config(config_path)
    download_openimages(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_openimages.yaml")
    args = parser.parse_args()
    main(args.config)
