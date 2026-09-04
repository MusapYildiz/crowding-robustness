"""
evaluate.py

Egitilmis modelleri crowding test seti uzerinde degerlendirir.

Metrikler:
  - Crowding Index (CI) = (Acc_baseline - Acc_crowding) / Acc_baseline
  - Spacing-CI egrisi
  - Per-class CI
  - Congruent vs Incongruent vs External karsilastirmasi
  - ECE (Expected Calibration Error)
  - Confusion matrix

Kullanim:
    python evaluate.py --config configs/config_openimages.yaml
    python evaluate.py --model resnet50 --config configs/config_openimages.yaml
"""

import json, argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader


MODEL_REGISTRY = {
    "resnet34":         ("resnet",  "resnet34"),
    "resnet50":         ("resnet",  "resnet50"),
    "resnet101":        ("resnet",  "resnet101"),
    "vit_s_16":         ("vit",    "vit_s_16"),
    "vit_b_16":         ("vit",    "vit_b_16"),
    "vgg_kagn11_v2":      ("convkan", "vgg_kagn11_v2"),
    "vgg_kagn11_v4":      ("convkan", "vgg_kagn11_v4"),
    "vgg_kagn_bn11sa_v4": ("convkan", "vgg_kagn_bn11sa_v4"),
}


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_dataset_classes(cfg: dict):
    dataset = cfg["data"].get("dataset", "coil100")
    if dataset == "openimages":
        from data.openimages.dataset import (IsolationDataset,
                                              CrowdingDataset,
                                              default_transform)
    elif dataset == "coco":
        from data.coco.dataset import (IsolationDataset,
                                        CrowdingDataset,
                                        default_transform)
    else:
        from data.dataset import (IsolationDataset,
                                   CrowdingDataset,
                                   default_transform)
    return IsolationDataset, CrowdingDataset, default_transform


def get_target_labels(cfg: dict):
    dataset = cfg["data"].get("dataset", "coil100")
    if dataset == "openimages":
        return cfg["data"]["target_classes"]
    elif dataset == "coco":
        return cfg["data"]["categories"]
    else:
        return [str(i) for i in cfg["data"]["target_obj_ids"]]


def load_model(model_name, num_classes, ckpt_path, device):
    from models.resnet  import build_resnet
    from models.vit     import build_vit
    from models.convkan import build_convkan

    family, variant = MODEL_REGISTRY[model_name]
    if family == "resnet":
        model = build_resnet(variant, num_classes, False, False)
    elif family == "vit":
        model = build_vit(variant, num_classes, False, False)
    elif family == "convkan":
        model = build_convkan(variant, num_classes, False, False)

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.no_grad()
def run_isolation(model, loader, device) -> dict:
    correct, correct_top5, total = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out  = model(imgs)
        correct      += (out.argmax(1) == labels).sum().item()
        top5          = out.topk(min(5, out.size(1)), dim=1).indices
        correct_top5 += (top5 == labels.unsqueeze(1)).any(1).sum().item()
        total += imgs.size(0)
    return {"top1_acc": round(correct/total, 4),
            "top5_acc": round(correct_top5/total, 4),
            "n_samples": total}


@torch.no_grad()
def run_crowding(model, loader, device) -> list:
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


def crowding_index(baseline_acc: float, crowded_acc: float) -> float:
    """CI = (Acc_baseline - Acc_crowding) / Acc_baseline"""
    if baseline_acc == 0:
        return 0.0
    return round((baseline_acc - crowded_acc) / baseline_acc, 4)


def spacing_ci_curve(records: list, baseline_acc: float) -> dict:
    """Her spacing seviyesi icin CI."""
    by_sp = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in records:
        k = r["spacing_deg"]
        by_sp[k]["correct"] += int(r["label"] == r["pred"])
        by_sp[k]["total"]   += 1
    curve = {}
    for k in sorted(by_sp):
        acc = round(by_sp[k]["correct"] / by_sp[k]["total"], 4)
        ci  = crowding_index(baseline_acc, acc)
        curve[str(k)] = {"accuracy": acc, "ci": ci}
    return curve


def flanker_type_ci(records: list, baseline_acc: float) -> dict:
    """Her flanker tipi icin CI."""
    by_ft = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in records:
        by_ft[r["flanker_type"]]["correct"] += int(r["label"] == r["pred"])
        by_ft[r["flanker_type"]]["total"]   += 1
    result = {}
    for k, v in by_ft.items():
        acc = round(v["correct"] / v["total"], 4)
        result[k] = {"accuracy": acc,
                     "ci": crowding_index(baseline_acc, acc)}
    return result


def per_class_ci(records: list, baseline_acc: float,
                  target_labels: list) -> dict:
    """Her sinif icin CI."""
    by_cls = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in records:
        c = r["category"]
        by_cls[c]["correct"] += int(r["label"] == r["pred"])
        by_cls[c]["total"]   += 1
    result = {}
    for label in target_labels:
        v   = by_cls.get(label, {"correct": 0, "total": 1})
        acc = round(v["correct"] / max(v["total"], 1), 4)
        result[label] = {"accuracy": acc,
                          "ci": crowding_index(baseline_acc, acc)}
    return result


def compute_ece(records: list, n_bins: int = 10) -> float:
    confs    = [max(r["prob"]) for r in records]
    corrects = [int(r["label"] == r["pred"]) for r in records]
    edges    = np.linspace(0, 1, n_bins + 1)
    ece, n   = 0.0, len(records)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i+1]
        idxs   = [j for j, c in enumerate(confs) if lo <= c < hi]
        if not idxs: continue
        avg_conf = np.mean([confs[j]    for j in idxs])
        avg_acc  = np.mean([corrects[j] for j in idxs])
        ece     += (len(idxs) / n) * abs(avg_conf - avg_acc)
    return round(float(ece), 4)


def confusion_matrix_fn(records: list, num_classes: int) -> list:
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for r in records:
        cm[r["label"]][r["pred"]] += 1
    return cm.tolist()


def evaluate_model(model_name: str, cfg: dict,
                   device: torch.device,
                   results_dir: Path) -> dict:
    print(f"\n{'='*60}\n  Degerlendirme: {model_name}\n{'='*60}")

    target_labels = get_target_labels(cfg)
    num_classes   = len(target_labels)
    dataset_dir   = cfg["data"]["output_dir"]
    composite_dir = str(Path(dataset_dir) / "composites")
    batch_size    = cfg["training"]["batch_size"]
    num_workers   = cfg["training"]["num_workers"]
    canvas_size   = cfg["image"]["canvas_size"]

    ckpt_path = results_dir / "checkpoints" / f"{model_name}_best.pth"
    if not ckpt_path.exists():
        print(f"  [SKIP] Checkpoint yok: {ckpt_path}")
        return {}

    IsolationDataset, CrowdingDataset, default_transform = \
        get_dataset_classes(cfg)

    model     = load_model(model_name, num_classes, ckpt_path, device)
    transform = default_transform(canvas_size)

    # Isolation baseline
    iso_ds  = IsolationDataset(dataset_dir, "test",
                                target_labels, transform)
    iso_loader = DataLoader(iso_ds, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)
    iso_m      = run_isolation(model, iso_loader, device)
    baseline   = iso_m["top1_acc"]
    print(f"  Isolation top-1: {baseline:.4f}")

    # Crowding test
    crowd_ds = CrowdingDataset(composite_dir, "test",
                                target_labels, transform=transform)
    crowd_loader = DataLoader(crowd_ds, batch_size=batch_size,
                               shuffle=False, num_workers=num_workers)
    records      = run_crowding(model, crowd_loader, device)

    if not records:
        print("  [WARN] Test kaydi yok.")
        return {}

    overall_acc = sum(r["label"] == r["pred"]
                      for r in records) / len(records)
    overall_ci  = crowding_index(baseline, overall_acc)

    results = {
        "model":              model_name,
        "isolation":          iso_m,
        "crowding_overall": {
            "accuracy": round(overall_acc, 4),
            "ci":       overall_ci,
        },
        "spacing_ci_curve":   spacing_ci_curve(records, baseline),
        "flanker_type_ci":    flanker_type_ci(records, baseline),
        "per_class_ci":       per_class_ci(records, baseline,
                                            target_labels),
        "ece":                compute_ece(records),
        "confusion_matrix":   confusion_matrix_fn(records, num_classes),
    }

    out_path = results_dir / f"{model_name}_eval.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"  CI={overall_ci:.4f}  ECE={results['ece']:.4f}")
    print(f"  Sonuclar: {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="all")
    parser.add_argument("--config", default="configs/config_openimages.yaml")
    args = parser.parse_args()

    cfg         = load_config(args.config)
    device      = torch.device(
        cfg["training"]["device"]
        if torch.cuda.is_available() else "cpu"
    )
    results_dir = Path(cfg["evaluation"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    models = (list(MODEL_REGISTRY.keys())
              if args.model == "all" else [args.model])

    all_results = []
    for m in models:
        res = evaluate_model(m, cfg, device, results_dir)
        if res: all_results.append(res)

    with open(results_dir / "evaluation_summary.json",
              "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nOzet kaydedildi.")


if __name__ == "__main__":
    main()
