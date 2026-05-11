"""
train.py

Tüm modelleri aynı protokolde eğitir.

Kullanım:
    python train.py --model resnet50 --config configs/config.yaml
    python train.py --model all     --config configs/config.yaml
"""

import os, time, json, argparse
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
    raise ValueError(f"Bilinmeyen aile: {family}")


def unfreeze_model(model: nn.Module, model_name: str, stage: int):
    family, _ = MODEL_REGISTRY[model_name]
    if family == "resnet":   unfreeze_resnet(model, stage)
    elif family == "vit":    unfreeze_vit(model, stage)
    elif family == "convkan": unfreeze_convkan(model, stage)


def count_parameters(model: nn.Module) -> dict:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def model_size_mb(model: nn.Module) -> float:
    total = sum(p.nbytes for p in model.parameters())
    total += sum(b.nbytes for b in model.buffers())
    return total / (1024 ** 2)


def measure_latency(model, device, canvas_size=224, n_runs=100) -> dict:
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


def measure_peak_gpu_ram(model, device, canvas_size, batch_size) -> float:
    if device.type != "cuda": return 0.0
    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    x = torch.randn(batch_size, 3, canvas_size, canvas_size).to(device)
    y = torch.zeros(batch_size, dtype=torch.long).to(device)
    out  = model(x)
    loss = nn.CrossEntropyLoss()(out, y)
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
    return round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 1)


def compute_flops(model, canvas_size) -> float:
    try:
        from fvcore.nn import FlopCountAnalysis
        dummy = torch.randn(1, 3, canvas_size, canvas_size)
        flops = FlopCountAnalysis(model.cpu(), dummy)
        flops.unsupported_ops_warnings(False)
        return round(flops.total() / 1e9, 2)
    except:
        return -1.0


def train_one_epoch(model, loader, optimizer, criterion, device) -> dict:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
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
        total_loss += loss.item() * imgs.size(0)
        correct    += (out.argmax(1) == labels).sum().item()
        top5        = out.topk(min(5, out.size(1)), dim=1).indices
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
    num_classes = len(cfg["data"]["target_obj_ids"])

    model = build_model(model_name, num_classes,
                        train_cfg["pretrained"], True).to(device)

    # Profil metrikleri
    params   = count_parameters(model)
    size_mb  = model_size_mb(model)
    flops_g  = compute_flops(model, canvas_size)
    latency  = measure_latency(model, device, canvas_size)
    peak_ram = measure_peak_gpu_ram(model, device, canvas_size,
                                    train_cfg["batch_size"])
    inf_ram  = measure_inference_ram(model, device, canvas_size)

    profile = {
        "model":                  model_name,
        "params_total":           params["total"],
        "model_size_mb":          round(size_mb, 1),
        "flops_gflops":           flops_g,
        "latency_ms":             latency["latency_ms"],
        "throughput_img_per_s":   latency["throughput_img_per_s"],
        "peak_gpu_ram_train_mb":  peak_ram,
        "inference_ram_mb":       inf_ram,
    }
    print(f"  Param: {params['total']:,} | "
          f"Boyut: {size_mb:.1f}MB | "
          f"FLOPs: {flops_g}G | "
          f"Latency: {latency['latency_ms']}ms")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                     lr=train_cfg["learning_rate"])
    scheduler = CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])

    warmup_end  = max(3,  train_cfg["epochs"] // 6)
    stage1_end  = max(6,  train_cfg["epochs"] // 3)
    stage2_end  = max(12, train_cfg["epochs"] * 2 // 3)

    best_val_acc, best_epoch, patience_cnt = 0.0, 0, 0
    history = []

    for epoch in range(1, train_cfg["epochs"] + 1):
        if epoch == warmup_end + 1:
            print(f"\n  [Epoch {epoch}] Unfreeze stage=1")
            unfreeze_model(model, model_name, 1)
            optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=train_cfg["learning_rate"] * 0.1)
            scheduler = CosineAnnealingLR(optimizer,
                                          T_max=train_cfg["epochs"] - epoch)
        elif epoch == stage1_end + 1:
            print(f"\n  [Epoch {epoch}] Unfreeze stage=2")
            unfreeze_model(model, model_name, 2)
            optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=train_cfg["learning_rate"] * 0.01)
            scheduler = CosineAnnealingLR(optimizer,
                                          T_max=train_cfg["epochs"] - epoch)
        elif epoch == stage2_end + 1:
            print(f"\n  [Epoch {epoch}] Unfreeze stage=3")
            unfreeze_model(model, model_name, 3)
            optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=train_cfg["learning_rate"] * 0.001)
            scheduler = CosineAnnealingLR(optimizer,
                                          T_max=train_cfg["epochs"] - epoch)

        train_m = train_one_epoch(model, loaders["train"],
                                   optimizer, criterion, device)
        val_m   = evaluate(model, loaders["val_isolation"], criterion, device)
        scheduler.step()

        history.append({
            "epoch":      epoch,
            "train_loss": train_m["loss"],
            "train_acc":  train_m["accuracy"],
            "val_loss":   val_m["loss"],
            "val_top1":   val_m["top1_acc"],
            "val_top5":   val_m["top5_acc"],
        })
        print(f"  Epoch {epoch:3d}/{train_cfg['epochs']}  "
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
            if patience_cnt >= train_cfg["early_stopping_patience"]:
                print(f"\n  Early stopping @ epoch {epoch} "
                      f"(best={best_val_acc:.4f} @ {best_epoch})")
                break

    threshold = best_val_acc * 0.90
    conv_epoch = next((r["epoch"] for r in history
                       if r["val_top1"] >= threshold), best_epoch)

    profile.update({
        "best_val_acc":      best_val_acc,
        "best_epoch":        best_epoch,
        "convergence_epoch": conv_epoch,
    })

    log = {"profile": profile, "history": history}
    with open(ckpt_dir / f"{model_name}_training_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n  Tamamlandı. best_val_acc={best_val_acc:.4f} @ epoch {best_epoch}")
    return profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="resnet50")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = torch.device(
        cfg["training"]["device"] if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    loaders  = get_dataloaders(cfg)
    ckpt_dir = Path(cfg["evaluation"]["results_dir"]) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    models_to_train = ALL_MODELS if args.model == "all" else [args.model]

    all_profiles = []
    for model_name in models_to_train:
        profile = train_model(model_name, cfg, loaders, device, ckpt_dir)
        all_profiles.append(profile)

    with open(Path(cfg["evaluation"]["results_dir"]) / "model_profiles.json", "w") as f:
        json.dump(all_profiles, f, indent=2)
    print("\nModel profilleri kaydedildi.")


if __name__ == "__main__":
    main()
