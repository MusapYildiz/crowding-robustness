"""
train.py

Tüm modelleri aynı protokolde eğitir.

Eğitim protokolü:
  1. Warm-up  (epoch 1..warmup_epochs)     : sadece classifier/head eğitilir
  2. Stage-1  (warmup..stage1_epochs)      : son bloklar açılır
  3. Stage-2  (stage1..stage2_epochs)      : daha derin bloklar açılır
  4. Full     (stage2..epochs)             : tüm model açılır

Kullanım:
    python train.py --model resnet50 --config configs/config.yaml
    python train.py --model all     --config configs/config.yaml
"""

import os
import time
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml

from data.dataset import get_dataloaders
from models.resnet  import build_resnet,  unfreeze_resnet
from models.vit     import build_vit,     unfreeze_vit
from models.convkan import build_convkan, unfreeze_convkan


# ── Model kayıt tablosu ─────────────────────────────────────────────────────
MODEL_REGISTRY = {
    # CNN
    "resnet34":   ("resnet",  "resnet34"),
    "resnet50":   ("resnet",  "resnet50"),
    "resnet101":  ("resnet",  "resnet101"),
    # Transformer
    "vit_s_16":   ("vit",    "vit_s_16"),
    "vit_b_16":   ("vit",    "vit_b_16"),
    # KAN
    "convkan_s":  ("convkan", "convkan_s"),
    "convkan_m":  ("convkan", "convkan_m"),
    "convkan_l":  ("convkan", "convkan_l"),
}

ALL_MODELS = list(MODEL_REGISTRY.keys())


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(model_name: str, num_classes: int,
                pretrained: bool, freeze_backbone: bool) -> nn.Module:
    family, variant = MODEL_REGISTRY[model_name]
    if family == "resnet":
        return build_resnet(variant, num_classes, pretrained, freeze_backbone)
    elif family == "vit":
        return build_vit(variant, num_classes, pretrained, freeze_backbone)
    elif family == "convkan":
        return build_convkan(variant, num_classes, pretrained, freeze_backbone)
    raise ValueError(f"Bilinmeyen model ailesi: {family}")


def unfreeze_model(model: nn.Module, model_name: str, stage: int) -> None:
    family, _ = MODEL_REGISTRY[model_name]
    if family == "resnet":
        unfreeze_resnet(model, stage)
    elif family == "vit":
        unfreeze_vit(model, stage)
    elif family == "convkan":
        unfreeze_convkan(model, stage)


def count_parameters(model: nn.Module) -> dict:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def model_size_mb(model: nn.Module) -> float:
    """Modelin parametre + buffer boyutunu MB cinsinden döndürür."""
    total_bytes = sum(p.nbytes for p in model.parameters())
    total_bytes += sum(b.nbytes for b in model.buffers())
    return total_bytes / (1024 ** 2)


def measure_latency(model: nn.Module, device: torch.device,
                    canvas_size: int = 224, n_runs: int = 100) -> dict:
    """
    Tek görsel inference latency ve throughput ölçer.
    GPU warm-up için ilk 10 run atlanır.
    """
    model.eval()
    dummy = torch.randn(1, 3, canvas_size, canvas_size).to(device)

    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    avg_ms = (sum(times) / len(times)) * 1000
    throughput = 1000 / avg_ms  # img/s
    return {"latency_ms": round(avg_ms, 3),
            "throughput_img_per_s": round(throughput, 1)}


def measure_peak_gpu_ram(model: nn.Module, device: torch.device,
                         canvas_size: int, batch_size: int) -> float:
    """Training sırasındaki peak GPU RAM kullanımını MB cinsinden ölçer."""
    if device.type != "cuda":
        return 0.0

    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    dummy_x = torch.randn(batch_size, 3, canvas_size, canvas_size).to(device)
    dummy_y = torch.zeros(batch_size, dtype=torch.long).to(device)

    criterion = nn.CrossEntropyLoss()
    out = model(dummy_x)
    loss = criterion(out, dummy_y)
    loss.backward()

    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    model.zero_grad()
    return round(peak, 1)


def measure_inference_ram(model: nn.Module, device: torch.device,
                          canvas_size: int) -> float:
    """Inference sırasındaki GPU RAM kullanımını MB cinsinden ölçer."""
    if device.type != "cuda":
        return 0.0

    torch.cuda.reset_peak_memory_stats(device)
    model.eval()
    dummy = torch.randn(1, 3, canvas_size, canvas_size).to(device)

    with torch.no_grad():
        _ = model(dummy)

    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return round(peak, 1)


def compute_flops(model: nn.Module, canvas_size: int) -> float:
    """FLOPs hesaplar (GFLOPs). fvcore gerektirir."""
    try:
        from fvcore.nn import FlopCountAnalysis
        dummy = torch.randn(1, 3, canvas_size, canvas_size)
        flops = FlopCountAnalysis(model.cpu(), dummy)
        flops.unsupported_ops_warnings(False)
        return round(flops.total() / 1e9, 2)  # GFLOPs
    except ImportError:
        return -1.0  # fvcore yoksa -1


def train_one_epoch(model: nn.Module, loader, optimizer: torch.optim.Optimizer,
                    criterion: nn.Module, device: torch.device) -> dict:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    return {
        "loss": round(total_loss / total, 4),
        "accuracy": round(correct / total, 4),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion: nn.Module,
             device: torch.device) -> dict:
    model.eval()
    total_loss, correct, correct_top5, total = 0.0, 0, 0, 0

    for batch in loader:
        # IsolationDataset: (img, label)
        # CrowdingDataset:  (img, label, meta)
        imgs, labels = batch[0].to(device), batch[1].to(device)

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()

        # Top-5
        top5 = outputs.topk(min(5, outputs.size(1)), dim=1).indices
        correct_top5 += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()
        total += imgs.size(0)

    return {
        "loss":     round(total_loss / total, 4),
        "top1_acc": round(correct / total, 4),
        "top5_acc": round(correct_top5 / total, 4),
    }


def train_model(model_name: str, cfg: dict,
                loaders: dict, device: torch.device,
                checkpoint_dir: Path) -> dict:
    """
    Tek bir modeli tam eğitim protokolüyle eğitir.
    Checkpoint ve training log kaydeder.
    """
    print(f"\n{'='*60}")
    print(f"  Model: {model_name}")
    print(f"{'='*60}")

    train_cfg  = cfg["training"]
    img_cfg    = cfg["image"]
    num_classes = len(cfg["data"]["categories"])

    # ── Model oluştur ───────────────────────────────────────────────────────
    model = build_model(
        model_name,
        num_classes=num_classes,
        pretrained=train_cfg["pretrained"],
        freeze_backbone=True,  # warm-up için dondur
    ).to(device)

    # ── Profil metrikleri (eğitim öncesi) ──────────────────────────────────
    params      = count_parameters(model)
    size_mb     = model_size_mb(model)
    flops_g     = compute_flops(model, img_cfg["canvas_size"])
    latency     = measure_latency(model, device, img_cfg["canvas_size"])
    peak_ram    = measure_peak_gpu_ram(model, device,
                                       img_cfg["canvas_size"],
                                       train_cfg["batch_size"])
    inf_ram     = measure_inference_ram(model, device, img_cfg["canvas_size"])

    profile = {
        "model":                   model_name,
        "params_total":            params["total"],
        "params_trainable_warmup": params["trainable"],
        "model_size_mb":           round(size_mb, 1),
        "flops_gflops":            flops_g,
        "latency_ms":              latency["latency_ms"],
        "throughput_img_per_s":    latency["throughput_img_per_s"],
        "peak_gpu_ram_train_mb":   peak_ram,
        "inference_ram_mb":        inf_ram,
    }
    print(f"  Parametre: {params['total']:,}  |  "
          f"Boyut: {size_mb:.1f}MB  |  "
          f"FLOPs: {flops_g}G  |  "
          f"Latency: {latency['latency_ms']}ms")

    # ── Optimizer & scheduler ───────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_cfg["learning_rate"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])

    # ── Unfreeze milestones ─────────────────────────────────────────────────
    # Epoch bazlı kademeli açma planı
    warmup_end  = max(3, train_cfg["epochs"] // 6)   # ilk ~5 epoch warm-up
    stage1_end  = max(6, train_cfg["epochs"] // 3)   # sonra stage-1
    stage2_end  = max(12, train_cfg["epochs"] * 2 // 3)  # sonra stage-2
    # stage2_end sonrası tüm model açılır

    # ── Eğitim döngüsü ─────────────────────────────────────────────────────
    best_val_acc = 0.0
    best_epoch   = 0
    patience_cnt = 0
    history      = []

    for epoch in range(1, train_cfg["epochs"] + 1):

        # Kademeli unfreeze
        if epoch == warmup_end + 1:
            print(f"\n  [Epoch {epoch}] Unfreeze stage=1")
            unfreeze_model(model, model_name, stage=1)
            optimizer = Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=train_cfg["learning_rate"] * 0.1,
            )
            scheduler = CosineAnnealingLR(
                optimizer, T_max=train_cfg["epochs"] - epoch
            )

        elif epoch == stage1_end + 1:
            print(f"\n  [Epoch {epoch}] Unfreeze stage=2")
            unfreeze_model(model, model_name, stage=2)
            optimizer = Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=train_cfg["learning_rate"] * 0.01,
            )
            scheduler = CosineAnnealingLR(
                optimizer, T_max=train_cfg["epochs"] - epoch
            )

        elif epoch == stage2_end + 1:
            print(f"\n  [Epoch {epoch}] Unfreeze stage=3 (tüm model)")
            unfreeze_model(model, model_name, stage=3)
            optimizer = Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=train_cfg["learning_rate"] * 0.001,
            )
            scheduler = CosineAnnealingLR(
                optimizer, T_max=train_cfg["epochs"] - epoch
            )

        # Train
        train_metrics = train_one_epoch(
            model, loaders["train"], optimizer, criterion, device
        )
        # Validation (isolation)
        val_metrics = evaluate(
            model, loaders["val_isolation"], criterion, device
        )

        scheduler.step()

        row = {
            "epoch":       epoch,
            "train_loss":  train_metrics["loss"],
            "train_acc":   train_metrics["accuracy"],
            "val_loss":    val_metrics["loss"],
            "val_top1":    val_metrics["top1_acc"],
            "val_top5":    val_metrics["top5_acc"],
        }
        history.append(row)

        print(f"  Epoch {epoch:3d}/{train_cfg['epochs']}  "
              f"train_acc={train_metrics['accuracy']:.4f}  "
              f"val_acc={val_metrics['top1_acc']:.4f}")

        # ── Best model & early stopping ─────────────────────────────────────
        if val_metrics["top1_acc"] > best_val_acc:
            best_val_acc = val_metrics["top1_acc"]
            best_epoch   = epoch
            patience_cnt = 0
            ckpt_path = checkpoint_dir / f"{model_name}_best.pth"
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_cnt += 1
            if patience_cnt >= train_cfg["early_stopping_patience"]:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(best val_acc={best_val_acc:.4f} @ epoch {best_epoch})")
                break

    # ── Convergence speed ───────────────────────────────────────────────────
    # Kaç epoch'ta %90 peak accuracy'ye ulaşıldı?
    threshold = best_val_acc * 0.90
    convergence_epoch = next(
        (r["epoch"] for r in history if r["val_top1"] >= threshold),
        best_epoch
    )

    profile["best_val_acc"]        = best_val_acc
    profile["best_epoch"]          = best_epoch
    profile["convergence_epoch"]   = convergence_epoch
    profile["params_trainable_full"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    # ── Log kaydet ──────────────────────────────────────────────────────────
    log = {"profile": profile, "history": history}
    log_path = checkpoint_dir / f"{model_name}_training_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n  Eğitim tamamlandı. Best val_acc={best_val_acc:.4f} "
          f"@ epoch {best_epoch}")
    print(f"  Log: {log_path}")

    return profile


def main():
    parser = argparse.ArgumentParser(description="Crowding benchmark eğitim")
    parser.add_argument("--model",  type=str, default="resnet50",
                        help=f"Model adı veya 'all'. Seçenekler: {ALL_MODELS}")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = torch.device(
        cfg["training"]["device"] if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    # DataLoader'lar
    loaders = get_dataloaders(cfg)

    # Checkpoint dizini
    ckpt_dir = Path(cfg["evaluation"]["results_dir"]) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Hangi modeller eğitilecek?
    if args.model == "all":
        models_to_train = ALL_MODELS
    elif args.model in MODEL_REGISTRY:
        models_to_train = [args.model]
    else:
        raise ValueError(f"Bilinmeyen model: {args.model}. "
                         f"Seçenekler: {ALL_MODELS} veya 'all'")

    # Eğitim
    all_profiles = []
    for model_name in models_to_train:
        profile = train_model(model_name, cfg, loaders, device, ckpt_dir)
        all_profiles.append(profile)

    # Tüm model profilleri özet olarak kaydet
    summary_path = Path(cfg["evaluation"]["results_dir"]) / "model_profiles.json"
    with open(summary_path, "w") as f:
        json.dump(all_profiles, f, indent=2)
    print(f"\nModel profilleri kaydedildi: {summary_path}")


if __name__ == "__main__":
    main()
