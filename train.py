"""
train.py

LP-FT stratejisiyle tum modelleri egitir.
(Kumar et al., ICLR 2022)

Asama 1 - Linear Probing (LP):
  Backbone dondurulur, sadece classifier egitilir.
  lr = lp_lr, epoch = lp_epochs

Asama 2 - Full Fine-Tuning (FT):
  Tum model aciilr, discriminative lr uygulanir.
  lr = ft_lr -> ft_lr_min (cosine decay), epoch = ft_epochs

Kullanim:
    python train.py --model resnet50 --config configs/config_openimages.yaml
    python train.py --model all     --config configs/config_openimages.yaml
"""

import os, time, json, argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml

from models.resnet  import build_resnet,  unfreeze_resnet
from models.vit     import build_vit,     unfreeze_vit
from models.convkan import build_convkan, unfreeze_convkan


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
ALL_MODELS = list(MODEL_REGISTRY.keys())


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_dataloaders(cfg: dict):
    dataset = cfg["data"].get("dataset", "coil100")
    if dataset == "openimages":
        from data.openimages.dataset import get_dataloaders as _get
    elif dataset == "coco":
        from data.coco.dataset import get_dataloaders as _get
    else:
        from data.dataset import get_dataloaders as _get
    return _get(cfg)


def get_num_classes(cfg: dict) -> int:
    dataset = cfg["data"].get("dataset", "coil100")
    if dataset == "openimages":
        return len(cfg["data"]["target_classes"])
    elif dataset == "coco":
        return len(cfg["data"]["categories"])
    else:
        return len(cfg["data"]["target_obj_ids"])


def build_model(model_name: str, num_classes: int,
                pretrained: bool,
                freeze_backbone: bool) -> nn.Module:
    family, variant = MODEL_REGISTRY[model_name]
    if family == "resnet":
        return build_resnet(variant, num_classes,
                             pretrained, freeze_backbone)
    elif family == "vit":
        return build_vit(variant, num_classes,
                          pretrained, freeze_backbone)
    elif family == "convkan":
        return build_convkan(variant, num_classes,
                              pretrained, freeze_backbone)
    raise ValueError(f"Bilinmeyen aile: {family}")


def model_size_mb(model: nn.Module) -> float:
    total = sum(p.nbytes for p in model.parameters())
    total += sum(b.nbytes for b in model.buffers())
    return total / (1024 ** 2)


def measure_latency(model, device, canvas_size=224,
                    n_runs=100) -> dict:
    model.eval()
    dummy = torch.randn(1, 3, canvas_size, canvas_size).to(device)
    with torch.no_grad():
        for _ in range(10): model(dummy)
    if device.type == "cuda": torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(dummy)
            if device.type == "cuda": torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    avg_ms = sum(times) / len(times) * 1000
    return {"latency_ms": round(avg_ms, 3),
            "throughput_img_per_s": round(1000 / avg_ms, 1)}


def measure_peak_gpu_ram(model, device, canvas_size,
                          batch_size) -> float:
    if device.type != "cuda": return 0.0
    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    x = torch.randn(batch_size, 3, canvas_size, canvas_size).to(device)
    y = torch.zeros(batch_size, dtype=torch.long).to(device)
    loss = nn.CrossEntropyLoss()(model(x), y)
    loss.backward()
    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    model.zero_grad()
    return round(peak, 1)


def measure_inference_ram(model, device, canvas_size) -> float:
    if device.type != "cuda": return 0.0
    torch.cuda.reset_peak_memory_stats(device)
    model.eval()
    dummy = torch.randn(1, 3, canvas_size, canvas_size).to(device)
    with torch.no_grad(): model(dummy)
    return round(torch.cuda.max_memory_allocated(device) / (1024**2), 1)


def compute_flops(model, canvas_size) -> float:
    try:
        from fvcore.nn import FlopCountAnalysis
        model_device = next(model.parameters()).device
        dummy  = torch.randn(1, 3, canvas_size, canvas_size)
        flops  = FlopCountAnalysis(model.cpu(), dummy)
        flops.unsupported_ops_warnings(False)
        result = round(flops.total() / 1e9, 2)
        model.to(model_device)
        return result
    except Exception:
        return -1.0


def train_one_epoch(model, loader, optimizer, criterion,
                    device, grad_clip) -> dict:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (out.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return {"loss": round(total_loss/total, 4),
            "accuracy": round(correct/total, 4)}


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> dict:
    model.eval()
    total_loss, correct, correct_top5, total = 0.0, 0, 0, 0
    for batch in loader:
        imgs, labels = batch[0].to(device), batch[1].to(device)
        out  = model(imgs)
        loss = criterion(out, labels)
        total_loss  += loss.item() * imgs.size(0)
        correct     += (out.argmax(1) == labels).sum().item()
        top5         = out.topk(min(5, out.size(1)), dim=1).indices
        correct_top5 += (top5 == labels.unsqueeze(1)).any(1).sum().item()
        total += imgs.size(0)
    return {"loss":     round(total_loss/total, 4),
            "top1_acc": round(correct/total, 4),
            "top5_acc": round(correct_top5/total, 4)}


def train_model(model_name: str, cfg: dict,
                loaders: dict, device: torch.device,
                ckpt_dir: Path) -> dict:
    print(f"\n{'='*60}\n  Model: {model_name}\n{'='*60}")

    train_cfg   = cfg["training"]
    canvas_size = cfg["image"]["canvas_size"]
    num_classes = get_num_classes(cfg)

    # ── Model ────────────────────────────────────────────────────
    model = build_model(model_name, num_classes,
                        train_cfg["pretrained"], True).to(device)

    # Profil metrikleri
    params   = sum(p.numel() for p in model.parameters())
    size_mb  = model_size_mb(model)
    flops_g  = compute_flops(model, canvas_size)
    latency  = measure_latency(model, device, canvas_size)
    peak_ram = measure_peak_gpu_ram(model, device, canvas_size,
                                    train_cfg["batch_size"])
    inf_ram  = measure_inference_ram(model, device, canvas_size)

    profile = {
        "model":                 model_name,
        "params_total":          params,
        "model_size_mb":         round(size_mb, 1),
        "flops_gflops":          flops_g,
        "latency_ms":            latency["latency_ms"],
        "throughput_img_per_s":  latency["throughput_img_per_s"],
        "peak_gpu_ram_train_mb": peak_ram,
        "inference_ram_mb":      inf_ram,
    }
    print(f"  Param: {params:,} | Boyut: {size_mb:.1f}MB | "
          f"FLOPs: {flops_g}G | Latency: {latency['latency_ms']}ms")

    criterion = nn.CrossEntropyLoss(
        label_smoothing=train_cfg.get("label_smoothing", 0.1)
    )
    grad_clip = train_cfg.get("gradient_clip", 1.0)
    patience  = train_cfg.get("early_stopping_patience", 7)

    best_val_acc, best_epoch, patience_cnt = 0.0, 0, 0
    history = []

    # ── ASAMA 1: Linear Probing ───────────────────────────────────
    print(f"\n  [LP] Linear Probing — {train_cfg['lp_epochs']} epoch")
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_cfg["lp_lr"],
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=train_cfg["lp_epochs"]
    )

    for epoch in range(1, train_cfg["lp_epochs"] + 1):
        train_m = train_one_epoch(model, loaders["train"],
                                   optimizer, criterion,
                                   device, grad_clip)
        val_m   = evaluate(model, loaders["val_isolation"],
                            criterion, device)
        scheduler.step()

        history.append({
            "epoch": epoch, "phase": "lp",
            "train_loss": train_m["loss"],
            "train_acc":  train_m["accuracy"],
            "val_top1":   val_m["top1_acc"],
            "val_top5":   val_m["top5_acc"],
        })
        print(f"  LP Epoch {epoch:3d}/{train_cfg['lp_epochs']}  "
              f"train={train_m['accuracy']:.4f}  "
              f"val={val_m['top1_acc']:.4f}")

        if val_m["top1_acc"] > best_val_acc:
            best_val_acc = val_m["top1_acc"]
            best_epoch   = epoch
            patience_cnt = 0
            torch.save(model.state_dict(),
                       ckpt_dir / f"{model_name}_best.pth")
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"  Early stopping (LP) @ epoch {epoch}")
                break

    # ── ASAMA 2: Full Fine-Tuning ─────────────────────────────────
    print(f"\n  [FT] Full Fine-Tuning — {train_cfg['ft_epochs']} epoch")

    # Tum modeli ac
    family, variant = MODEL_REGISTRY[model_name]
    if family == "resnet":
        unfreeze_resnet(model, stage=3)
    elif family == "vit":
        unfreeze_vit(model, stage=3)
    elif family == "convkan":
        unfreeze_convkan(model, stage=1)

    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg["ft_lr"],
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=train_cfg["ft_epochs"],
        eta_min=train_cfg.get("ft_lr_min", 1e-6),
    )

    patience_cnt = 0

    for epoch in range(1, train_cfg["ft_epochs"] + 1):
        train_m = train_one_epoch(model, loaders["train"],
                                   optimizer, criterion,
                                   device, grad_clip)
        val_m   = evaluate(model, loaders["val_isolation"],
                            criterion, device)
        scheduler.step()

        global_epoch = train_cfg["lp_epochs"] + epoch
        history.append({
            "epoch": global_epoch, "phase": "ft",
            "train_loss": train_m["loss"],
            "train_acc":  train_m["accuracy"],
            "val_top1":   val_m["top1_acc"],
            "val_top5":   val_m["top5_acc"],
        })
        print(f"  FT Epoch {epoch:3d}/{train_cfg['ft_epochs']}  "
              f"train={train_m['accuracy']:.4f}  "
              f"val={val_m['top1_acc']:.4f}")

        if val_m["top1_acc"] > best_val_acc:
            best_val_acc = val_m["top1_acc"]
            best_epoch   = global_epoch
            patience_cnt = 0
            torch.save(model.state_dict(),
                       ckpt_dir / f"{model_name}_best.pth")
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"  Early stopping (FT) @ epoch {epoch}")
                break

    profile.update({
        "best_val_acc": best_val_acc,
        "best_epoch":   best_epoch,
    })

    log = {"profile": profile, "history": history}
    with open(ckpt_dir / f"{model_name}_training_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n  Tamamlandi. best_val_acc={best_val_acc:.4f} "
          f"@ epoch {best_epoch}")
    return profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="resnet50")
    parser.add_argument("--config", default="configs/config_openimages.yaml")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = torch.device(
        cfg["training"]["device"]
        if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    loaders  = get_dataloaders(cfg)
    ckpt_dir = Path(cfg["evaluation"]["results_dir"]) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    models_to_train = (ALL_MODELS if args.model == "all"
                       else [args.model])

    all_profiles = []
    for model_name in models_to_train:
        profile = train_model(model_name, cfg, loaders,
                               device, ckpt_dir)
        all_profiles.append(profile)

    # Mevcut profilleri oku, guncelle veya ekle
    profiles_path = Path(cfg["evaluation"]["results_dir"]) / "model_profiles.json"
    existing = []
    if profiles_path.exists():
        with open(profiles_path) as f:
            existing = json.load(f)

    # Ayni model varsa guncelle, yoksa ekle
    existing_names = {p["model"]: i for i, p in enumerate(existing)}
    for profile in all_profiles:
        if profile["model"] in existing_names:
            existing[existing_names[profile["model"]]] = profile
        else:
            existing.append(profile)

    with open(profiles_path, "w") as f:
        json.dump(existing, f, indent=2)
    print("\nModel profilleri kaydedildi.")


if __name__ == "__main__":
    main()
