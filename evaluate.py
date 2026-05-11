"""
evaluate.py

Eğitilmiş modelleri crowding test seti üzerinde değerlendirir.

Kullanım:
    python evaluate.py --config configs/config.yaml
    python evaluate.py --model resnet50 --config configs/config.yaml
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

def _get_dataset_classes(cfg):
    dataset = cfg["data"].get("dataset", "coil100")
    if dataset == "coco":
        from data.coco.dataset import IsolationDataset, CrowdingDataset, default_transform
    else:
        from data.dataset import IsolationDataset, CrowdingDataset, default_transform
    return IsolationDataset, CrowdingDataset, default_transform
from models.resnet  import build_resnet
from models.vit     import build_vit
from models.convkan import build_convkan


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


def load_config(path): 
    with open(path) as f: return yaml.safe_load(f)


def load_model(model_name, num_classes, ckpt_path, device):
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
        total        += imgs.size(0)
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
                "obj_id":       metas["obj_id"][i] if isinstance(metas["obj_id"][i], int)
                                else metas["obj_id"][i].item(),
                "flanker_type": metas["flanker_type"][i],
                "spacing_deg":  float(metas["spacing_deg"][i]),
            })
    return records


def accuracy_drop(baseline, crowded):
    return round((baseline - crowded) / baseline, 4) if baseline > 0 else 0.0


def spacing_accuracy_curve(records):
    by_sp = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in records:
        k = r["spacing_deg"]
        by_sp[k]["correct"] += int(r["label"] == r["pred"])
        by_sp[k]["total"]   += 1
    return {str(k): round(v["correct"]/v["total"], 4)
            for k, v in sorted(by_sp.items())}


def flanker_type_effect(records):
    by_ft = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in records:
        by_ft[r["flanker_type"]]["correct"] += int(r["label"] == r["pred"])
        by_ft[r["flanker_type"]]["total"]   += 1
    return {k: round(v["correct"]/v["total"], 4) for k, v in by_ft.items()}


def per_class_accuracy(records, target_ids):
    by_cls = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in records:
        by_cls[r["obj_id"]]["correct"] += int(r["label"] == r["pred"])
        by_cls[r["obj_id"]]["total"]   += 1
    return {str(obj): round(by_cls[obj]["correct"]/max(by_cls[obj]["total"],1), 4)
            for obj in target_ids}


def critical_spacing(sp_curve, baseline_acc, drop_threshold=0.5):
    target   = baseline_acc * (1 - drop_threshold)
    spacings = sorted(float(k) for k in sp_curve.keys())
    for sp in spacings:
        if sp_curve[str(sp)] <= target:
            return sp
    return spacings[-1]


def robustness_index(crit_sp, object_size_deg):
    return round(crit_sp / object_size_deg, 3) if object_size_deg > 0 else 0.0


def compute_ece(records, n_bins=10):
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


def confusion_matrix(records, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for r in records:
        cm[r["label"]][r["pred"]] += 1
    return cm.tolist()


def evaluate_model(model_name, cfg, device, results_dir):
    print(f"\n{'='*60}\n  Değerlendirme: {model_name}\n{'='*60}")

    dataset_type  = cfg["data"].get("dataset", "coil100")
    if dataset_type == "coco":
        target_ids  = cfg["data"]["categories"]
        num_classes = len(target_ids)
    else:
        target_ids  = cfg["data"]["target_obj_ids"]
        num_classes = len(target_ids)
    dataset_dir   = cfg["data"]["output_dir"]
    composite_dir = str(Path(dataset_dir) / "composites")
    batch_size    = cfg["training"]["batch_size"]
    num_workers   = cfg["training"]["num_workers"]
    canvas_size   = cfg["image"]["canvas_size"]
    object_size   = cfg["image"]["object_size"]
    pix_per_deg   = cfg["spacing"]["pixels_per_degree"]

    ckpt_path = results_dir / "checkpoints" / f"{model_name}_best.pth"
    if not ckpt_path.exists():
        print(f"  [SKIP] Checkpoint yok: {ckpt_path}")
        return {}

    model     = load_model(model_name, num_classes, ckpt_path, device)
    transform = default_transform(canvas_size)

    # Isolation baseline
    IsolationDataset, CrowdingDataset, default_transform = _get_dataset_classes(cfg)
    iso_ds  = IsolationDataset(dataset_dir, "test", target_ids, transform)
    iso_loader = DataLoader(iso_ds, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    iso_m    = run_isolation(model, iso_loader, device)
    baseline = iso_m["top1_acc"]
    print(f"  Isolation top-1: {baseline:.4f}")

    # Crowding test
    crowd_ds = CrowdingDataset(composite_dir, "test", target_ids,
                               transform=transform)
    crowd_loader = DataLoader(crowd_ds, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    records      = run_crowding(model, crowd_loader, device)
    overall_acc  = sum(r["label"] == r["pred"] for r in records) / len(records)

    sp_curve        = spacing_accuracy_curve(records)
    ft_effect       = flanker_type_effect(records)
    pc_acc          = per_class_accuracy(records, target_ids)
    ece             = compute_ece(records)
    cm              = confusion_matrix(records, num_classes)
    object_size_deg = object_size / pix_per_deg
    crit_sp         = critical_spacing(sp_curve, baseline)
    rob_idx         = robustness_index(crit_sp, object_size_deg)

    results = {
        "model":                   model_name,
        "isolation":               iso_m,
        "crowding_overall_acc":    round(overall_acc, 4),
        "accuracy_drop":           accuracy_drop(baseline, overall_acc),
        "spacing_accuracy_curve":  sp_curve,
        "flanker_type_effect":     ft_effect,
        "per_class_accuracy":      pc_acc,
        "critical_spacing_deg":    crit_sp,
        "robustness_index":        rob_idx,
        "ece":                     ece,
        "confusion_matrix":        cm,
        "object_size_deg":         round(object_size_deg, 3),
    }

    out_path = results_dir / f"{model_name}_eval.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"  accuracy_drop={results['accuracy_drop']:.4f}  "
          f"critical_spacing={crit_sp}°  ECE={ece:.4f}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="all")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg         = load_config(args.config)
    device      = torch.device(
        cfg["training"]["device"] if torch.cuda.is_available() else "cpu"
    )
    results_dir = Path(cfg["evaluation"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    models = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]

    all_results = []
    for m in models:
        res = evaluate_model(m, cfg, device, results_dir)
        if res: all_results.append(res)

    with open(results_dir / "evaluation_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nÖzet kaydedildi: {results_dir}/evaluation_summary.json")


if __name__ == "__main__":
    main()
