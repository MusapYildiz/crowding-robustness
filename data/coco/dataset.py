"""
data/coco/dataset.py

COCO icin PyTorch Dataset siniflari.

  IsolationDataset : train/val -- tek nesne gorselleri
  CrowdingDataset  : test      -- hedef + flanker kompozitler
"""

import json
from pathlib import Path
from typing import Optional, Callable

import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


def default_transform(image_size: int = 224) -> Callable:
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


class IsolationDataset(Dataset):
    """
    Train ve validation icin.
    dataset_dir/split/category/*.png
    """

    def __init__(self, dataset_dir: str, split: str,
                 categories: list,
                 transform: Optional[Callable] = None):
        self.dataset_dir = Path(dataset_dir)
        self.split       = split
        self.categories  = sorted(categories)
        self.class_to_idx = {c: i for i, c in enumerate(self.categories)}
        self.transform   = transform or default_transform()
        self.samples     = self._load_samples()

    def _load_samples(self) -> list:
        samples = []
        for cat in self.categories:
            cat_dir = self.dataset_dir / self.split / cat
            if not cat_dir.exists():
                continue
            for p in sorted(cat_dir.glob("*.png")):
                samples.append({
                    "path":     p,
                    "label":    self.class_to_idx[cat],
                    "category": cat,
                })
        return samples

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        img = cv2.imread(str(s["path"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        return img, s["label"]

    def get_class_counts(self) -> dict:
        counts = {c: 0 for c in self.categories}
        for s in self.samples:
            counts[s["category"]] += 1
        return counts


class CrowdingDataset(Dataset):
    """
    Test icin.
    composites/split/category/flanker_type/deg/*.png
    """

    def __init__(self, composite_dir: str, split: str,
                 categories: list,
                 flanker_types: Optional[list] = None,
                 spacing_degrees: Optional[list] = None,
                 transform: Optional[Callable] = None):
        self.composite_dir   = Path(composite_dir) / split
        self.categories      = sorted(categories)
        self.class_to_idx    = {c: i for i, c in enumerate(self.categories)}
        self.flanker_types   = flanker_types
        self.spacing_degrees = spacing_degrees
        self.transform       = transform or default_transform()
        self.samples         = self._load_samples()

    def _load_samples(self) -> list:
        samples = []
        for cat in self.categories:
            cat_dir = self.composite_dir / cat
            if not cat_dir.exists():
                continue
            for ft_dir in sorted(cat_dir.iterdir()):
                ft = ft_dir.name
                if self.flanker_types and ft not in self.flanker_types:
                    continue
                for sp_dir in sorted(ft_dir.iterdir()):
                    deg = float(sp_dir.name.replace("deg", ""))
                    if (self.spacing_degrees and
                            deg not in self.spacing_degrees):
                        continue
                    for p in sorted(sp_dir.glob("*.png")):
                        samples.append({
                            "path":         p,
                            "label":        self.class_to_idx[cat],
                            "category":     cat,
                            "flanker_type": ft,
                            "spacing_deg":  deg,
                        })
        return samples

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        img = cv2.imread(str(s["path"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        meta = {
            "category":     s["category"],
            "flanker_type": s["flanker_type"],
            "spacing_deg":  s["spacing_deg"],
            "label":        s["label"],
        }
        return img, s["label"], meta


def get_dataloaders(cfg: dict) -> dict:
    dataset_dir   = cfg["data"]["output_dir"]
    composite_dir = str(Path(dataset_dir) / "composites")
    categories    = cfg["data"]["categories"]
    batch_size    = cfg["training"]["batch_size"]
    num_workers   = cfg["training"]["num_workers"]
    transform     = default_transform(cfg["image"]["canvas_size"])

    train_ds     = IsolationDataset(dataset_dir, "train",
                                    categories, transform)
    val_iso_ds   = IsolationDataset(dataset_dir, "val",
                                    categories, transform)
    val_crowd_ds = CrowdingDataset(composite_dir, "val",
                                   categories, transform=transform)
    test_ds      = CrowdingDataset(composite_dir, "test",
                                   categories, transform=transform)

    return {
        "train":         DataLoader(train_ds,     batch_size=batch_size,
                                    shuffle=True,  num_workers=num_workers,
                                    pin_memory=True),
        "val_isolation": DataLoader(val_iso_ds,   batch_size=batch_size,
                                    shuffle=False, num_workers=num_workers,
                                    pin_memory=True),
        "val_crowding":  DataLoader(val_crowd_ds, batch_size=batch_size,
                                    shuffle=False, num_workers=num_workers,
                                    pin_memory=True),
        "test":          DataLoader(test_ds,      batch_size=batch_size,
                                    shuffle=False, num_workers=num_workers,
                                    pin_memory=True),
    }
