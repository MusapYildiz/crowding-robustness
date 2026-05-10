"""
evaluate.py

Eğitilmiş modelleri crowding test seti üzerinde değerlendirir.

Hesaplanan metrikler:
  - Top-1 / Top-5 accuracy (isolation baseline)
  - Accuracy drop (baseline → crowded)
  - Spacing-accuracy eğrisi (her spacing seviyesi için)
  - Flanker tipi etkisi (same_class / different_class / external)
  - Per-class accuracy
  - Critical spacing (accuracy'nin %50 düştüğü spacing)
  - Calibration (ECE — Expected Calibration Error)
  - Confusion matrix
  - Robustness index (Bouma uyumu)

Kullanım:
    python evaluate.py --config configs/config.yaml
    python evaluate.py --model resnet50 --config configs/config.yaml
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from data.dataset import IsolationDataset, CrowdingDataset, default_transform
from torch.utils.data import DataLoader
from models.resnet  import build_resnet
from models.vit     import build_vit
from models.convkan import build_convkan


# ── Model yükleme ───────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "resnet34":  ("resnet",  "resnet34"),
    "resnet50":  ("resnet",  "resnet50"),
    "resnet101": ("resnet",  "resnet101"),
    "vit_s_16":  ("vit",    "vit_s_16"),
    "vit_b_16":  ("vit",    "vit_b_16"),
    "convkan_s": ("convkan", "convkan_s"),
    "convkan_m": ("convkan", "convkan_m"),
    "convkan_l": ("convkan", "convkan_l"),
}


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(model_name: str, num_classes: int,
               checkpoint_path: Path, device: torch.device) -> nn.Module:
    family, variant = MODEL_REGISTRY[model_name]

    if family == "resnet":
        model = build_resnet(variant, num_classes, pretrained=False,
                             freeze_backbone=False)
    elif family == "vit":
        model = build_vit(variant, num_classes, pretrained=False,
                          freeze_backbone=False)
    elif family == "convkan":
        model = build_convkan(variant, num_classes, pretrained=False,
                              freeze_backbone=False)
    else:
        raise ValueError(f"Bilinmeyen aile: {family}")

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ── Temel metrikler ─────────────────────────────────────────────────────────

@torch.no_grad()
def run_isolation(model: nn.Module, loader: DataLoader,
                  device: torch.device) -> dict:
    """Isolation (baseline) accuracy — flanker yok."""
    correct, correct_top5, total = 0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)

        correct += (out.argmax(1) == labels).sum().item()
        top5 = out.topk(min(5, out.size(1)), dim=1).indices
        correct_top5 += (top5 == labels.unsqueeze(1)).any(1).sum().item()
        total += imgs.size(0)

    return {
        "top1_acc": round(correct / total, 4),
        "top5_acc": round(correct_top5 / total, 4),
        "n_samples": total,
    }


@torch.no_grad()
def run_crowding(model: nn.Module, loader: DataLoader,
                 device: torch.device) -> list:
    """
    Crowding test seti üzerinde her örnek için tahmin toplar.
    Döndürür: [{label, pred, prob, category, flanker_type, spacing_deg}, ...]
    """
    records = []

    for batch in loader:
        imgs, labels, metas = batch
        imgs, labels = imgs.to(device), labels.to(device)
        out  = model(imgs)
        prob = F.softmax(out, dim=1)
        pred = out.argmax(1)

        for i in range(imgs.size(0)):
            records.append({
                "label":        labels[i].item(),
                "pred":         pred[i].item(),
                "prob":         prob[i].cpu().tolist(),
                "category":     metas["category"][i],
                "flanker_type": metas["flanker_type"][i],
                "spacing_deg":  float(metas["spacing_deg"][i]),
            })

    return records


# ── Crowding metrikleri ─────────────────────────────────────────────────────

def accuracy_drop(baseline_acc: float, crowded_acc: float) -> float:
    """Relative accuracy drop."""
    if baseline_acc == 0:
        return 0.0
    return round((baseline_acc - crowded_acc) / baseline_acc, 4)


def spacing_accuracy_curve(records: list) -> dict:
    """Her spacing seviyesi için accuracy."""
    by_spacing = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in records:
        key = r["spacing_deg"]
        by_spacing[key]["correct"] += int(r["label"] == r["pred"])
        by_spacing[key]["total"]   += 1

    return {
        str(k): round(v["correct"] / v["total"], 4)
        for k, v in sorted(by_spacing.items())
    }


def flanker_type_effect(records: list) -> dict:
    """Her flanker tipi için accuracy."""
    by_type = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in records:
        ft = r["flanker_type"]
        by_type[ft]["correct"] += int(r["label"] == r["pred"])
        by_type[ft]["total"]   += 1

    return {
        k: round(v["correct"] / v["total"], 4)
        for k, v in by_type.items()
    }


def per_class_accuracy(records: list, categories: list) -> dict:
    """Her kategori için accuracy."""
    by_class = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in records:
        cat = r["category"]
        by_class[cat]["correct"] += int(r["label"] == r["pred"])
        by_class[cat]["total"]   += 1

    return {
        cat: round(by_class[cat]["correct"] / max(by_class[cat]["total"], 1), 4)
        for cat in categories
    }


def critical_spacing(spacing_curve: dict, baseline_acc: float,
                     drop_threshold: float = 0.5) -> float:
    """
    Accuracy'nin baseline'ın (1 - drop_threshold) katına düştüğü spacing.
    Bouma'nın kritik spacing tanımıyla uyumlu.
    drop_threshold=0.5 → accuracy baseline'ın %50'sine düştüğünde.
    """
    target = baseline_acc * (1 - drop_threshold)
    spacings = sorted(float(k) for k in spacing_curve.keys())

    for sp in spacings:
        if spacing_curve[str(sp)] <= target:
            return sp
    return spacings[-1]  # hiç bu kadar düşmüyorsa en büyük spacing


def robustness_index(crit_spacing: float, object_size_deg: float) -> float:
    """
    Bouma uyumu: b = critical_spacing / eccentricity
    Biz eksentriklik yerine nesne boyutunu (derece) kullanıyoruz.
    Bouma sabitine (b ≈ 0.4-0.5) ne kadar yakın?
    """
    if object_size_deg == 0:
        return 0.0
    return round(crit_spacing / object_size_deg, 3)


def compute_ece(records: list, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE).
    Modelin güven skorlarının ne kadar kalibre edildiğini ölçer.
    ECE ≈ 0 → mükemmel kalibrasyon.
    """
    confidences = [max(r["prob"]) for r in records]
    corrects    = [int(r["label"] == r["pred"]) for r in records]

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n   = len(records)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        idxs = [j for j, c in enumerate(confidences) if lo <= c < hi]
        if not idxs:
            continue
        avg_conf = np.mean([confidences[j] for j in idxs])
        avg_acc  = np.mean([corrects[j]    for j in idxs])
        ece += (len(idxs) / n) * abs(avg_conf - avg_acc)

    return round(float(ece), 4)


def confusion_matrix(records: list, num_classes: int) -> list:
    """Confusion matrix (num_classes × num_classes)."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for r in records:
        cm[r["label"]][r["pred"]] += 1
    return cm.tolist()


# ── Ana değerlendirme fonksiyonu ────────────────────────────────────────────

def evaluate_model(model_name: str, cfg: dict,
                   device: torch.device,
                   results_dir: Path) -> dict:
    print(f"\n{'='*60}")
    print(f"  Değerlendirme: {model_name}")
    print(f"{'='*60}")

    categories  = cfg["data"]["categories"]
    num_classes = len(categories)
    dataset_dir = cfg["data"]["output_dir"]
    composite_dir = str(Path(dataset_dir) / "composites")
    batch_size  = cfg["training"]["batch_size"]
    num_workers = cfg["training"]["num_workers"]
    canvas_size = cfg["image"]["canvas_size"]
    object_size = cfg["image"]["object_size"]
    pix_per_deg = cfg["spacing"]["pixels_per_degree"]

    # Checkpoint
    ckpt_path = results_dir / "checkpoints" / f"{model_name}_best.pth"
    if not ckpt_path.exists():
        print(f"  [SKIP] Checkpoint bulunamadı: {ckpt_path}")
        return {}

    model = load_model(model_name, num_classes, ckpt_path, device)
    transform = default_transform(canvas_size)

    # ── Isolation baseline ──────────────────────────────────────────────────
    iso_ds = IsolationDataset(dataset_dir, "test", categories, transform)
    iso_loader = DataLoader(iso_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    iso_metrics = run_isolation(model, iso_loader, device)
    baseline_acc = iso_metrics["top1_acc"]
    print(f"  Isolation baseline top-1: {baseline_acc:.4f}")

    # ── Crowding test ───────────────────────────────────────────────────────
    crowd_ds = CrowdingDataset(composite_dir, "test", categories,
                               transform=transform)
    crowd_loader = DataLoader(crowd_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)
    records = run_crowding(model, crowd_loader, device)

    overall_acc = sum(r["label"] == r["pred"] for r in records) / len(records)

    # ── Metrikler ───────────────────────────────────────────────────────────
    sp_curve  = spacing_accuracy_curve(records)
    ft_effect = flanker_type_effect(records)
    pc_acc    = per_class_accuracy(records, categories)
    ece       = compute_ece(records)
    cm        = confusion_matrix(records, num_classes)

    # Nesne boyutu derece cinsinden
    object_size_deg = object_size / pix_per_deg

    crit_sp = critical_spacing(sp_curve, baseline_acc, drop_threshold=0.5)
    rob_idx = robustness_index(crit_sp, object_size_deg)

    results = {
        "model":                  model_name,
        "isolation": {
            "top1_acc":           iso_metrics["top1_acc"],
            "top5_acc":           iso_metrics["top5_acc"],
            "n_samples":          iso_metrics["n_samples"],
        },
        "crowding_overall_acc":   round(overall_acc, 4),
        "accuracy_drop":          accuracy_drop(baseline_acc, overall_acc),
        "spacing_accuracy_curve": sp_curve,
        "flanker_type_effect":    ft_effect,
        "per_class_accuracy":     pc_acc,
        "critical_spacing_deg":   crit_sp,
        "robustness_index":       rob_idx,
        "ece":                    ece,
        "confusion_matrix":       cm,
        "object_size_deg":        round(object_size_deg, 3),
    }

    # Kaydet
    out_path = results_dir / f"{model_name}_eval.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Sonuçlar kaydedildi: {out_path}")

    print(f"  accuracy_drop={results['accuracy_drop']:.4f}  "
          f"critical_spacing={crit_sp}°  "
          f"ECE={ece:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Crowding benchmark değerlendirme")
    parser.add_argument("--model",  type=str, default="all")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = torch.device(
        cfg["training"]["device"] if torch.cuda.is_available() else "cpu"
    )
    results_dir = Path(cfg["evaluation"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    models_to_eval = (
        list(MODEL_REGISTRY.keys()) if args.model == "all"
        else [args.model]
    )

    all_results = []
    for model_name in models_to_eval:
        res = evaluate_model(model_name, cfg, device, results_dir)
        if res:
            all_results.append(res)

    # Özet tablo
    summary_path = results_dir / "evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nÖzet kaydedildi: {summary_path}")


if __name__ == "__main__":
    main()
