"""
dataset.py

COIL-100 için PyTorch Dataset sınıfları.

  IsolationDataset : train/val — tek nesne görselleri
  CrowdingDataset  : test     — hedef + flanker kompozitler
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
    Train ve validation için.
    dataset_dir/split/obj{id}/*.png
    """

    def __init__(self, dataset_dir: str, split: str,
                 target_obj_ids: list,
                 transform: Optional[Callable] = None):
        self.dataset_dir    = Path(dataset_dir)
        self.split          = split
        self.target_obj_ids = sorted(target_obj_ids)
        self.transform      = transform or default_transform()
        self.obj_to_idx     = {obj: i for i, obj in enumerate(self.target_obj_ids)}
        self.samples        = self._load_samples()

    def _load_samples(self) -> list:
        samples = []
        for obj_id in self.target_obj_ids:
            obj_dir = self.dataset_dir / self.split / f"obj{obj_id}"
            if not obj_dir.exists():
                continue
            for p in sorted(obj_dir.glob("*.png")):
                samples.append({"path": p, "label": self.obj_to_idx[obj_id],
                                 "obj_id": obj_id})
        return samples

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        img = cv2.imread(str(s["path"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        return img, s["label"]

    def get_class_counts(self) -> dict:
        counts = {obj_id: 0 for obj_id in self.target_obj_ids}
        for s in self.samples:
            counts[s["obj_id"]] += 1
        return counts


class CrowdingDataset(Dataset):
    """
    Test için.
    composites/split/obj{id}/{flanker_type}/{deg}deg/*.png
    """

    def __init__(self, composite_dir: str, split: str,
                 target_obj_ids: list,
                 flanker_types: Optional[list] = None,
                 spacing_degrees: Optional[list] = None,
                 transform: Optional[Callable] = None):
        self.composite_dir  = Path(composite_dir) / split
        self.target_obj_ids = sorted(target_obj_ids)
        self.obj_to_idx     = {obj: i for i, obj in enumerate(self.target_obj_ids)}
        self.flanker_types  = flanker_types
        self.spacing_degrees = spacing_degrees
        self.transform      = transform or default_transform()
        self.samples        = self._load_samples()

    def _load_samples(self) -> list:
        samples = []
        for obj_id in self.target_obj_ids:
            obj_dir = self.composite_dir / f"obj{obj_id}"
            if not obj_dir.exists():
                continue
            for ft_dir in sorted(obj_dir.iterdir()):
                ft = ft_dir.name
                if self.flanker_types and ft not in self.flanker_types:
                    continue
                for sp_dir in sorted(ft_dir.iterdir()):
                    deg = float(sp_dir.name.replace("deg", ""))
                    if self.spacing_degrees and deg not in self.spacing_degrees:
                        continue
                    for p in sorted(sp_dir.glob("*.png")):
                        samples.append({
                            "path":         p,
                            "label":        self.obj_to_idx[obj_id],
                            "obj_id":       obj_id,
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
            "obj_id":       s["obj_id"],
            "flanker_type": s["flanker_type"],
            "spacing_deg":  s["spacing_deg"],
            "label":        s["label"],
            "category":     str(s["obj_id"]),
        }
        return img, s["label"], meta


def get_dataloaders(cfg: dict) -> dict:
    dataset_dir   = cfg["data"]["output_dir"]
    composite_dir = str(Path(dataset_dir) / "composites")
    target_ids    = cfg["data"]["target_obj_ids"]
    batch_size    = cfg["training"]["batch_size"]
    num_workers   = cfg["training"]["num_workers"]
    transform     = default_transform(cfg["image"]["canvas_size"])

    train_ds    = IsolationDataset(dataset_dir, "train", target_ids, transform)
    val_iso_ds  = IsolationDataset(dataset_dir, "val",   target_ids, transform)
    val_crowd_ds = CrowdingDataset(composite_dir, "val", target_ids,
                                   transform=transform)
    test_ds     = CrowdingDataset(composite_dir, "test", target_ids,
                                   transform=transform)

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
