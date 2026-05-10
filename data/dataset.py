"""
dataset.py

PyTorch Dataset sınıfları:
  - IsolationDataset : train ve val için tek nesne görselleri
  - CrowdingDataset  : test için kompozit (hedef + flanker) görseller
"""

import json
from pathlib import Path
from typing import Optional, Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


def default_transform(image_size: int = 224) -> Callable:
    """
    Standart ImageNet normalize transform.
    Pretrained modeller bu normalize'a alışkın.
    """
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


class IsolationDataset(Dataset):
    """
    Train ve validation için kullanılır.
    Her görsel: uniform background üzerinde tek nesne.

    Dizin yapısı:
        dataset_dir/
          train/
            elephant/  *.png
            giraffe/   *.png
            ...
    """

    def __init__(self, dataset_dir: str, split: str,
                 categories: list, transform: Optional[Callable] = None):
        """
        Args:
            dataset_dir : prepare_dataset.py'nin çıktı dizini
            split       : "train" veya "val"
            categories  : kullanılacak kategori listesi
            transform   : torchvision transform (None ise default kullanılır)
        """
        self.dataset_dir = Path(dataset_dir)
        self.split       = split
        self.categories  = categories
        self.transform   = transform or default_transform()

        self.class_to_idx = {cat: i for i, cat in enumerate(sorted(categories))}
        self.samples = self._load_samples()

    def _load_samples(self) -> list:
        samples = []
        for cat in self.categories:
            cat_dir = self.dataset_dir / self.split / cat
            if not cat_dir.exists():
                print(f"[WARN] Dizin bulunamadı: {cat_dir}")
                continue
            for img_path in sorted(cat_dir.glob("*.png")):
                samples.append({
                    "path":  img_path,
                    "label": self.class_to_idx[cat],
                    "category": cat,
                })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        sample = self.samples[idx]
        img = cv2.imread(str(sample["path"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        return img, sample["label"]

    def get_class_counts(self) -> dict:
        counts = {cat: 0 for cat in self.categories}
        for s in self.samples:
            counts[s["category"]] += 1
        return counts


class CrowdingDataset(Dataset):
    """
    Test için kullanılır.
    Her görsel: hedef nesne + sol/sağ flanker (belirli spacing ve tip).

    Dizin yapısı:
        composites/
          test/
            elephant/
              same_class/
                1deg/  *.png
                2deg/  *.png
              different_class/
                ...
              external/
                ...
    """

    def __init__(self, composite_dir: str, split: str,
                 categories: list,
                 flanker_types: Optional[list] = None,
                 spacing_degrees: Optional[list] = None,
                 transform: Optional[Callable] = None):
        """
        Args:
            composite_dir   : build_composites.py'nin çıktı dizini
            split           : "test" veya "val"
            categories      : hedef kategoriler
            flanker_types   : filtre (None = hepsi)
            spacing_degrees : filtre (None = hepsi)
            transform       : torchvision transform
        """
        self.composite_dir   = Path(composite_dir) / split
        self.split           = split
        self.categories      = categories
        self.flanker_types   = flanker_types
        self.spacing_degrees = spacing_degrees
        self.transform       = transform or default_transform()

        self.class_to_idx = {cat: i for i, cat in enumerate(sorted(categories))}
        self.samples = self._load_samples()

    def _load_samples(self) -> list:
        samples = []
        for cat in self.categories:
            cat_dir = self.composite_dir / cat
            if not cat_dir.exists():
                continue

            for flanker_type_dir in sorted(cat_dir.iterdir()):
                flanker_type = flanker_type_dir.name
                if self.flanker_types and flanker_type not in self.flanker_types:
                    continue

                for spacing_dir in sorted(flanker_type_dir.iterdir()):
                    deg = float(spacing_dir.name.replace("deg", ""))
                    if self.spacing_degrees and deg not in self.spacing_degrees:
                        continue

                    for img_path in sorted(spacing_dir.glob("*.png")):
                        samples.append({
                            "path":         img_path,
                            "label":        self.class_to_idx[cat],
                            "category":     cat,
                            "flanker_type": flanker_type,
                            "spacing_deg":  deg,
                        })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        sample = self.samples[idx]
        img = cv2.imread(str(sample["path"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        meta = {
            "category":     sample["category"],
            "flanker_type": sample["flanker_type"],
            "spacing_deg":  sample["spacing_deg"],
            "label":        sample["label"],
        }
        return img, sample["label"], meta

    def get_conditions(self) -> list:
        """Mevcut (flanker_type, spacing_deg) kombinasyonlarını döndürür."""
        conds = set()
        for s in self.samples:
            conds.add((s["flanker_type"], s["spacing_deg"]))
        return sorted(conds)


def get_dataloaders(cfg: dict) -> dict:
    """
    Config'e göre tüm DataLoader'ları oluşturur ve döndürür.

    Returns:
        {
            "train": DataLoader,
            "val_isolation": DataLoader,  # izolasyon val (baseline)
            "val_crowding": DataLoader,   # flanker'lı val
            "test": DataLoader,           # flanker'lı test (tüm koşullar)
        }
    """
    from torch.utils.data import DataLoader

    dataset_dir   = cfg["data"]["output_dir"]
    composite_dir = str(Path(dataset_dir) / "composites")
    categories    = cfg["data"]["categories"]
    batch_size    = cfg["training"]["batch_size"]
    num_workers   = cfg["training"]["num_workers"]
    transform     = default_transform(cfg["image"]["canvas_size"])

    train_ds = IsolationDataset(dataset_dir, "train", categories, transform)
    val_iso_ds = IsolationDataset(dataset_dir, "val", categories, transform)
    val_crowd_ds = CrowdingDataset(composite_dir, "val", categories,
                                   transform=transform)
    test_ds = CrowdingDataset(composite_dir, "test", categories,
                              transform=transform)

    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True),
        "val_isolation": DataLoader(val_iso_ds, batch_size=batch_size,
                                    shuffle=False, num_workers=num_workers,
                                    pin_memory=True),
        "val_crowding": DataLoader(val_crowd_ds, batch_size=batch_size,
                                   shuffle=False, num_workers=num_workers,
                                   pin_memory=True),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True),
    }
