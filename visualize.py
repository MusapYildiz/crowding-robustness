"""
visualize.py

Değerlendirme sonuçlarından tüm grafikleri ve Grad-CAM görsellerini üretir.

Grafikler:
  1. Spacing-accuracy eğrisi (tüm modeller, karşılaştırmalı)
  2. Accuracy drop bar chart (model × flanker tipi)
  3. Per-class accuracy heatmap
  4. Confusion matrix (her model için)
  5. Model profil tablosu (parametre, FLOPs, latency, RAM...)
  6. Calibration (reliability diagram, ECE)
  7. Robustness index karşılaştırması
  8. Grad-CAM görselleştirme (isolation vs crowded)

Kullanım:
    python visualize.py --config configs/config.yaml
    python visualize.py --gradcam --model resnet50 --config configs/config.yaml
"""

import json
import argparse
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import yaml

# Grad-CAM için
import torch
import torch.nn.functional as F
import cv2


# ── Stil sabitleri ───────────────────────────────────────────────────────────
PALETTE = {
    "resnet34":  "#4E79A7",
    "resnet50":  "#2563EB",
    "resnet101": "#1E3A8A",
    "vit_s_16":  "#F28E2B",
    "vit_b_16":  "#B45309",
    "convkan_s": "#59A14F",
    "convkan_m": "#16A34A",
    "convkan_l": "#166534",
}

FAMILY_LABELS = {
    "resnet34":  "ResNet-34 (CNN)",
    "resnet50":  "ResNet-50 (CNN)",
    "resnet101": "ResNet-101 (CNN)",
    "vit_s_16":  "ViT-S/16 (Transformer)",
    "vit_b_16":  "ViT-B/16 (Transformer)",
    "convkan_s": "ConvKAN-S (KAN)",
    "convkan_m": "ConvKAN-M (KAN)",
    "convkan_l": "ConvKAN-L (KAN)",
}

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":     150,
})


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_eval_results(results_dir: Path) -> list:
    results = []
    for p in sorted(results_dir.glob("*_eval.json")):
        with open(p) as f:
            results.append(json.load(f))
    return results


def load_profiles(results_dir: Path) -> list:
    path = results_dir / "model_profiles.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


# ── 1. Spacing-accuracy eğrisi ───────────────────────────────────────────────

def plot_spacing_accuracy(results: list, out_dir: Path):
    fig, ax = plt.subplots(figsize=(9, 5))

    for res in results:
        model = res["model"]
        curve = res["spacing_accuracy_curve"]
        baseline = res["isolation"]["top1_acc"]

        spacings = sorted(float(k) for k in curve.keys())
        accs     = [curve[str(s)] for s in spacings]

        # Baseline çizgisi (noktalı)
        ax.axhline(baseline, color=PALETTE.get(model, "gray"),
                   linestyle="--", alpha=0.35, linewidth=0.8)

        ax.plot(spacings, accs,
                label=FAMILY_LABELS.get(model, model),
                color=PALETTE.get(model, "gray"),
                marker="o", linewidth=2, markersize=5)

    ax.set_xlabel("Flanker Spacing (°)")
    ax.set_ylabel("Top-1 Accuracy")
    ax.set_title("Spacing–Accuracy Eğrisi\n(noktalı: isolation baseline)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = out_dir / "spacing_accuracy_curve.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Kaydedildi: {path}")


# ── 2. Accuracy drop — flanker tipi × model ─────────────────────────────────

def plot_accuracy_drop(results: list, out_dir: Path):
    rows = []
    for res in results:
        model    = res["model"]
        baseline = res["isolation"]["top1_acc"]
        for ft, acc in res["flanker_type_effect"].items():
            drop = round((baseline - acc) / baseline, 4) if baseline > 0 else 0
            rows.append({
                "Model":        FAMILY_LABELS.get(model, model),
                "Flanker Tipi": ft,
                "Accuracy Drop": drop,
                "_color":       PALETTE.get(model, "gray"),
            })

    df = pd.DataFrame(rows)
    flanker_order = ["same_class", "different_class", "external"]
    flanker_labels = {
        "same_class":       "Aynı Sınıf",
        "different_class":  "Farklı Sınıf",
        "external":         "Harici",
    }
    df["Flanker Tipi"] = df["Flanker Tipi"].map(flanker_labels)

    fig, ax = plt.subplots(figsize=(11, 5))
    models = df["Model"].unique()
    x = np.arange(len(flanker_order))
    width = 0.8 / len(models)

    for i, model_label in enumerate(models):
        sub = df[df["Model"] == model_label]
        model_key = [k for k, v in FAMILY_LABELS.items() if v == model_label]
        color = PALETTE.get(model_key[0], "gray") if model_key else "gray"
        vals = [
            sub[sub["Flanker Tipi"] == flanker_labels[ft]]["Accuracy Drop"].values
            for ft in flanker_order
        ]
        vals = [v[0] if len(v) > 0 else 0 for v in vals]
        offset = (i - len(models) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width=width * 0.9,
               label=model_label, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([flanker_labels[f] for f in flanker_order])
    ax.set_ylabel("Relative Accuracy Drop")
    ax.set_title("Flanker Tipine Göre Accuracy Drop")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = out_dir / "accuracy_drop_by_flanker.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Kaydedildi: {path}")


# ── 3. Per-class accuracy heatmap ────────────────────────────────────────────

def plot_per_class_heatmap(results: list, out_dir: Path):
    models     = [FAMILY_LABELS.get(r["model"], r["model"]) for r in results]
    categories = list(results[0]["per_class_accuracy"].keys()) if results else []

    matrix = np.array([
        [r["per_class_accuracy"].get(cat, 0) for cat in categories]
        for r in results
    ])

    fig, ax = plt.subplots(figsize=(max(8, len(categories)), max(4, len(models) * 0.6)))
    sns.heatmap(matrix, annot=True, fmt=".2f",
                xticklabels=categories, yticklabels=models,
                cmap="RdYlGn", vmin=0, vmax=1, ax=ax,
                linewidths=0.3, linecolor="white")
    ax.set_title("Per-Class Accuracy (Crowding Test)")
    ax.set_xlabel("Kategori")
    ax.set_ylabel("Model")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    path = out_dir / "per_class_accuracy_heatmap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Kaydedildi: {path}")


# ── 4. Confusion matrix ──────────────────────────────────────────────────────

def plot_confusion_matrices(results: list, categories: list, out_dir: Path):
    n = len(results)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 5, rows * 4.5))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for i, res in enumerate(results):
        cm = np.array(res["confusion_matrix"])
        # Normalize per row
        row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
        cm_norm  = cm / row_sums

        sns.heatmap(cm_norm, annot=True, fmt=".2f",
                    xticklabels=categories, yticklabels=categories,
                    cmap="Blues", vmin=0, vmax=1, ax=axes[i],
                    linewidths=0.2, cbar=False)
        axes[i].set_title(FAMILY_LABELS.get(res["model"], res["model"]),
                          fontsize=9)
        axes[i].set_xlabel("Tahmin", fontsize=8)
        axes[i].set_ylabel("Gerçek", fontsize=8)
        axes[i].tick_params(labelsize=7)
        plt.setp(axes[i].get_xticklabels(), rotation=30, ha="right")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Confusion Matrix (Normalized, Crowding Test)", fontsize=12)
    plt.tight_layout()
    path = out_dir / "confusion_matrices.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Kaydedildi: {path}")


# ── 5. Model profil tablosu ──────────────────────────────────────────────────

def plot_model_profile_table(profiles: list, out_dir: Path):
    if not profiles:
        print("  [SKIP] Model profil verisi yok.")
        return

    cols = [
        ("model",                  "Model"),
        ("params_total",           "Parametre"),
        ("model_size_mb",          "Boyut (MB)"),
        ("flops_gflops",           "FLOPs (G)"),
        ("latency_ms",             "Latency (ms)"),
        ("throughput_img_per_s",   "Throughput (img/s)"),
        ("peak_gpu_ram_train_mb",  "Peak GPU RAM\nTrain (MB)"),
        ("inference_ram_mb",       "Inference\nRAM (MB)"),
        ("best_val_acc",           "Best Val\nAcc"),
        ("convergence_epoch",      "Conv.\nEpoch"),
    ]

    rows = []
    for p in profiles:
        row = []
        for key, _ in cols:
            val = p.get(key, "—")
            if key == "params_total" and isinstance(val, int):
                val = f"{val/1e6:.1f}M"
            row.append(val)
        rows.append(row)

    fig, ax = plt.subplots(figsize=(16, max(3, len(profiles) * 0.7 + 1.5)))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=[label for _, label in cols],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.6)

    # Renklendirme: başlık satırı
    for j in range(len(cols)):
        table[0, j].set_facecolor("#1E3A8A")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Aile rengi satırlara
    for i, p in enumerate(profiles):
        color = PALETTE.get(p.get("model", ""), "#F5F5F5")
        for j in range(len(cols)):
            table[i + 1, j].set_facecolor(color + "33")  # %20 opaklık

    ax.set_title("Model Profil Tablosu", fontsize=13, pad=10)
    plt.tight_layout()
    path = out_dir / "model_profile_table.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Kaydedildi: {path}")


# ── 6. Calibration — reliability diagram ────────────────────────────────────

def plot_calibration(results: list, out_dir: Path):
    """
    ECE bar chart — her modelin kalibrasyon hatasını gösterir.
    """
    models = [FAMILY_LABELS.get(r["model"], r["model"]) for r in results]
    eces   = [r.get("ece", 0) for r in results]
    colors = [PALETTE.get(r["model"], "gray") for r in results]

    fig, ax = plt.subplots(figsize=(max(7, len(results) * 1.2), 4))
    bars = ax.bar(models, eces, color=colors, alpha=0.85, width=0.6)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    ax.set_ylabel("ECE (↓ daha iyi)")
    ax.set_title("Expected Calibration Error (Crowding Test)")
    plt.xticks(rotation=20, ha="right")
    ax.set_ylim(0, max(eces) * 1.3 if eces else 0.5)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = out_dir / "calibration_ece.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Kaydedildi: {path}")


# ── 7. Robustness index ──────────────────────────────────────────────────────

def plot_robustness_index(results: list, out_dir: Path):
    models  = [FAMILY_LABELS.get(r["model"], r["model"]) for r in results]
    ri      = [r.get("robustness_index", 0) for r in results]
    crit_sp = [r.get("critical_spacing_deg", 0) for r in results]
    colors  = [PALETTE.get(r["model"], "gray") for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Critical spacing
    bars = axes[0].bar(models, crit_sp, color=colors, alpha=0.85, width=0.6)
    axes[0].bar_label(bars, fmt="%.1f°", padding=3, fontsize=8)
    axes[0].axhline(2.0, color="red", linestyle="--", linewidth=1,
                    label="Bouma kritik sınır (~2°)")
    axes[0].set_ylabel("Critical Spacing (°)")
    axes[0].set_title("Critical Spacing")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)
    plt.setp(axes[0].get_xticklabels(), rotation=20, ha="right")

    # Robustness index
    bars2 = axes[1].bar(models, ri, color=colors, alpha=0.85, width=0.6)
    axes[1].bar_label(bars2, fmt="%.2f", padding=3, fontsize=8)
    axes[1].axhspan(0.4, 0.5, alpha=0.1, color="green",
                    label="Bouma sabiti aralığı (0.4–0.5)")
    axes[1].set_ylabel("Robustness Index (b)")
    axes[1].set_title("Robustness Index (Bouma Uyumu)")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)
    plt.setp(axes[1].get_xticklabels(), rotation=20, ha="right")

    plt.tight_layout()
    path = out_dir / "robustness_index.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Kaydedildi: {path}")


# ── 8. Grad-CAM ──────────────────────────────────────────────────────────────

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    Hedef katmana hook bağlar, backward pass sonrası ısı haritası üretir.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(_, __, output):
            self.activations = output.detach()

        def backward_hook(_, __, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor,
                 class_idx: int = None) -> np.ndarray:
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        # Global average pooling over gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalize
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam


def apply_colormap(cam: np.ndarray, img_rgb: np.ndarray) -> np.ndarray:
    """CAM'i görsel üzerine ısı haritası olarak uygular."""
    h, w = img_rgb.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(
        (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.5 * img_rgb + 0.5 * heatmap).astype(np.uint8)
    return overlay


def plot_gradcam(model_name: str, cfg: dict,
                 results_dir: Path, out_dir: Path,
                 n_samples: int = 4):
    """
    Her model için isolation ve crowded görseller üzerinde Grad-CAM üretir.
    n_samples: kaç örnek görseli üretilsin.
    """
    from data.dataset import IsolationDataset, CrowdingDataset, default_transform
    from torch.utils.data import DataLoader
    from models.resnet  import build_resnet,  get_resnet_gradcam_layer
    from models.vit     import build_vit,     get_vit_gradcam_layer
    from models.convkan import build_convkan

    categories  = cfg["data"]["categories"]
    num_classes = len(categories)
    dataset_dir = cfg["data"]["output_dir"]
    composite_dir = str(Path(dataset_dir) / "composites")
    canvas_size = cfg["image"]["canvas_size"]

    ckpt_path = results_dir / "checkpoints" / f"{model_name}_best.pth"
    if not ckpt_path.exists():
        print(f"  [SKIP] Checkpoint yok: {ckpt_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model + Grad-CAM katmanı
    family = model_name.split("_")[0] if "_" in model_name else model_name.rstrip("0123456789")
    if "resnet" in model_name:
        model       = build_resnet(model_name, num_classes, False, False)
        target_layer = get_resnet_gradcam_layer(model)
    elif "vit" in model_name:
        model       = build_vit(model_name, num_classes, False, False)
        target_layer = get_vit_gradcam_layer(model)
    elif "convkan" in model_name:
        model       = build_convkan(model_name, num_classes, False, False)
        target_layer = model.get_gradcam_layer()
    else:
        print(f"  [SKIP] Grad-CAM desteklenmiyor: {model_name}")
        return

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)

    grad_cam = GradCAM(model, target_layer)
    transform = default_transform(canvas_size)

    # Isolation örnekleri
    iso_ds = IsolationDataset(dataset_dir, "test", categories, transform)
    # Crowding örnekleri (same_class, 2deg)
    crowd_ds = CrowdingDataset(composite_dir, "test", categories,
                               flanker_types=["same_class"],
                               spacing_degrees=[2.0],
                               transform=transform)

    fig, axes = plt.subplots(n_samples, 4,
                             figsize=(14, n_samples * 3.5))

    for i in range(n_samples):
        if i >= len(iso_ds) or i >= len(crowd_ds):
            break

        # Isolation
        iso_img_t, iso_label = iso_ds[i]
        iso_img_t = iso_img_t.unsqueeze(0).to(device)

        # Crowding
        crowd_img_t, crowd_label, _ = crowd_ds[i]
        crowd_img_t = crowd_img_t.unsqueeze(0).to(device)

        # Orijinal görsel (denormalize)
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])

        def to_rgb(t):
            img = t.squeeze().cpu().numpy().transpose(1, 2, 0)
            img = (img * std + mean).clip(0, 1)
            return (img * 255).astype(np.uint8)

        iso_rgb   = to_rgb(iso_img_t)
        crowd_rgb = to_rgb(crowd_img_t)

        # Grad-CAM
        iso_cam   = grad_cam.generate(iso_img_t)
        crowd_cam = grad_cam.generate(crowd_img_t)

        iso_overlay   = apply_colormap(iso_cam,   iso_rgb)
        crowd_overlay = apply_colormap(crowd_cam, crowd_rgb)

        # Çiz
        for j, (img, title) in enumerate([
            (iso_rgb,       f"Isolation\n({categories[iso_label]})"),
            (iso_overlay,   "Grad-CAM\n(Isolation)"),
            (crowd_rgb,     f"Crowded (2°)\n({categories[iso_label]})"),
            (crowd_overlay, "Grad-CAM\n(Crowded)"),
        ]):
            axes[i][j].imshow(img)
            axes[i][j].set_title(title, fontsize=8)
            axes[i][j].axis("off")

    fig.suptitle(
        f"Grad-CAM: {FAMILY_LABELS.get(model_name, model_name)}\n"
        f"(Model flanker varken nereye bakıyor?)",
        fontsize=11
    )
    plt.tight_layout()
    path = out_dir / f"gradcam_{model_name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Kaydedildi: {path}")


# ── Ana fonksiyon ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Crowding benchmark görselleştirme")
    parser.add_argument("--config",  type=str, default="configs/config.yaml")
    parser.add_argument("--gradcam", action="store_true",
                        help="Grad-CAM görsellerini de üret")
    parser.add_argument("--model",   type=str, default="all",
                        help="Grad-CAM için model adı")
    args = parser.parse_args()

    cfg         = load_config(args.config)
    results_dir = Path(cfg["evaluation"]["results_dir"])
    out_dir     = results_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    results  = load_eval_results(results_dir)
    profiles = load_profiles(results_dir)
    categories = cfg["data"]["categories"]

    if not results:
        print("Değerlendirme sonucu bulunamadı. Önce evaluate.py çalıştırın.")
        return

    print(f"\n{len(results)} model sonucu yüklendi. Grafikler üretiliyor...\n")

    plot_spacing_accuracy(results, out_dir)
    plot_accuracy_drop(results, out_dir)
    plot_per_class_heatmap(results, out_dir)
    plot_confusion_matrices(results, categories, out_dir)
    plot_model_profile_table(profiles, out_dir)
    plot_calibration(results, out_dir)
    plot_robustness_index(results, out_dir)

    if args.gradcam:
        from models.resnet import MODEL_REGISTRY as RMAP
        models_for_gradcam = (
            list(FAMILY_LABELS.keys()) if args.model == "all"
            else [args.model]
        )
        print("\nGrad-CAM üretiliyor...")
        for model_name in models_for_gradcam:
            print(f"  {model_name}...")
            plot_gradcam(model_name, cfg, results_dir, out_dir)

    print(f"\nTüm grafikler kaydedildi: {out_dir}")


if __name__ == "__main__":
    main()
