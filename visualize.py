"""
visualize.py

Degerlendirme sonuclarindan grafikler uretir.

Grafikler:
  1. Spacing-accuracy egrisi
  2. Crowding Index bar chart (flanker tipine gore)
  3. Per-class CI heatmap
  4. Confusion matrix
  5. Model profil tablosu
  6. ECE bar chart
  7. Robustness index (varsa)

Kullanim:
    python visualize.py --config configs/config_openimages.yaml
"""

import json
import argparse
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import yaml


# ── Stil sabitleri ────────────────────────────────────────────────────────────
PALETTE = {
    "resnet34":         "#4E79A7",
    "resnet50":         "#2563EB",
    "resnet101":        "#1E3A8A",
    "vit_s_16":         "#F28E2B",
    "vit_b_16":         "#B45309",
    "vgg_kagn11_v2":      "#86EFAC",
    "vgg_kagn11_v4":      "#22C55E",
    "vgg_kagn_bn11sa_v4": "#166534",
}

LABELS = {
    "resnet34":         "ResNet-34 (CNN)",
    "resnet50":         "ResNet-50 (CNN)",
    "resnet101":        "ResNet-101 (CNN)",
    "vit_s_16":         "ViT-S/16 (Transformer)",
    "vit_b_16":         "ViT-B/16 (Transformer)",
    "vgg_kagn11_v2":      "VGG KAGN-11v2 (KAN)",
    "vgg_kagn11_v4":      "VGG KAGN-11v4 (KAN)",
    "vgg_kagn_bn11sa_v4": "VGG KAGN-BN-SA-11v4 (KAN)",
}

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "figure.dpi":       150,
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


def get_categories(cfg: dict) -> list:
    dataset = cfg["data"].get("dataset", "coil100")
    if dataset == "openimages":
        return cfg["data"]["target_classes"]
    elif dataset == "coco":
        return cfg["data"]["categories"]
    else:
        return [str(i) for i in cfg["data"]["target_obj_ids"]]


def _get_accuracy(val) -> float:
    """Hem dict hem float formatini destekler."""
    if isinstance(val, dict):
        return float(val.get("accuracy", 0))
    return float(val) if val is not None else 0.0


def _get_ci(val) -> float:
    """CI degerini al."""
    if isinstance(val, dict):
        return float(val.get("ci", 0))
    return 0.0


def _overall_ci(res: dict) -> float:
    co = res.get("crowding_overall", {})
    if isinstance(co, dict):
        return float(co.get("ci", 0))
    return _get_ci(res.get("accuracy_drop", 0))


# ── 1. Spacing-Accuracy Egrisi ────────────────────────────────────────────────
def plot_spacing_accuracy(results: list, out_dir: Path):
    fig, ax = plt.subplots(figsize=(9, 5))

    for res in results:
        model    = res["model"]
        curve    = res.get("spacing_ci_curve",
                           res.get("spacing_accuracy_curve", {}))
        baseline = res["isolation"]["top1_acc"]

        spacings = sorted(float(k) for k in curve.keys())
        accs     = [_get_accuracy(curve[str(s)]) for s in spacings]

        ax.axhline(baseline,
                   color=PALETTE.get(model, "gray"),
                   linestyle="--", alpha=0.35, linewidth=0.8)
        ax.plot(spacings, accs,
                label=LABELS.get(model, model),
                color=PALETTE.get(model, "gray"),
                marker="o", linewidth=2, markersize=5)

    ax.set_xlabel("Flanker Spacing (deg)")
    ax.set_ylabel("Top-1 Accuracy")
    ax.set_title("Spacing-Accuracy Egrisi\n(noktalı: isolation baseline)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = out_dir / "spacing_accuracy_curve.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Kaydedildi: {path}")


# ── 2. Crowding Index Bar Chart ───────────────────────────────────────────────
def plot_crowding_index(results: list, out_dir: Path):
    rows = []
    for res in results:
        model    = res["model"]
        baseline = res["isolation"]["top1_acc"]
        ft_data  = res.get("flanker_type_ci",
                           res.get("flanker_type_effect", {}))

        for ft, val in ft_data.items():
            acc  = _get_accuracy(val)
            ci   = round((baseline - acc) / baseline, 4) if baseline > 0 else 0
            rows.append({
                "Model":       LABELS.get(model, model),
                "FlankerTipi": ft,
                "CI":          ci,
                "_color":      PALETTE.get(model, "gray"),
            })

    if not rows:
        return

    df = pd.DataFrame(rows)
    flanker_types  = sorted(df["FlankerTipi"].unique())
    models         = df["Model"].unique()
    x     = np.arange(len(flanker_types))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, model_label in enumerate(models):
        sub    = df[df["Model"] == model_label]
        model_key = next((k for k, v in LABELS.items() if v == model_label), None)
        color  = PALETTE.get(model_key, "gray") if model_key else "gray"
        vals   = [sub[sub["FlankerTipi"] == ft]["CI"].values[0]
                  if len(sub[sub["FlankerTipi"] == ft]) > 0 else 0
                  for ft in flanker_types]
        offset = (i - len(models) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width=width * 0.9,
               label=model_label, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(flanker_types)
    ax.set_ylabel("Crowding Index (CI)")
    ax.set_title("Flanker Tipine Gore Crowding Index")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = out_dir / "crowding_index_by_flanker.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Kaydedildi: {path}")


# ── 3. Per-Class CI Heatmap ───────────────────────────────────────────────────
def plot_per_class_heatmap(results: list, categories: list, out_dir: Path):
    models = [LABELS.get(r["model"], r["model"]) for r in results]
    pc_key = "per_class_ci" if "per_class_ci" in results[0] else "per_class_accuracy"

    matrix = np.array([
        [_get_accuracy(r[pc_key].get(cat, 0)) for cat in categories]
        for r in results
    ])

    fig, ax = plt.subplots(
        figsize=(max(8, len(categories) * 0.7), max(4, len(models) * 0.6))
    )
    sns.heatmap(matrix, annot=True, fmt=".2f",
                xticklabels=categories, yticklabels=models,
                cmap="RdYlGn", vmin=0, vmax=1, ax=ax,
                linewidths=0.3, linecolor="white")
    ax.set_title("Per-Class Accuracy (Crowding Test)")
    ax.set_xlabel("Kategori")
    ax.set_ylabel("Model")
    plt.xticks(rotation=35, ha="right", fontsize=8)
    plt.tight_layout()
    path = out_dir / "per_class_accuracy_heatmap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Kaydedildi: {path}")


# ── 4. Confusion Matrix ───────────────────────────────────────────────────────
def plot_confusion_matrices(results: list, categories: list, out_dir: Path):
    n    = len(results)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                              figsize=(cols * 5, rows * 4.5))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for i, res in enumerate(results):
        cm = np.array(res.get("confusion_matrix", []))
        if cm.size == 0:
            axes[i].axis("off")
            continue
        row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
        cm_norm  = cm / row_sums

        sns.heatmap(cm_norm, annot=False,
                    xticklabels=categories, yticklabels=categories,
                    cmap="Blues", vmin=0, vmax=1, ax=axes[i],
                    linewidths=0.1, cbar=False)
        axes[i].set_title(LABELS.get(res["model"], res["model"]), fontsize=9)
        axes[i].set_xlabel("Tahmin", fontsize=7)
        axes[i].set_ylabel("Gercek", fontsize=7)
        axes[i].tick_params(labelsize=6)
        plt.setp(axes[i].get_xticklabels(), rotation=35, ha="right")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Confusion Matrix (Normalized)", fontsize=12)
    plt.tight_layout()
    path = out_dir / "confusion_matrices.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Kaydedildi: {path}")


# ── 5. Model Profil Tablosu ───────────────────────────────────────────────────
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
        ("throughput_img_per_s",   "Throughput"),
        ("peak_gpu_ram_train_mb",  "Peak GPU\nRAM (MB)"),
        ("inference_ram_mb",       "Inf. RAM\n(MB)"),
        ("best_val_acc",           "Best Val\nAcc"),
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

    fig, ax = plt.subplots(
        figsize=(16, max(3, len(profiles) * 0.7 + 1.5))
    )
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=[label for _, label in cols],
        cellLoc="center", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.6)
    for j in range(len(cols)):
        table[0, j].set_facecolor("#1E3A8A")
        table[0, j].set_text_props(color="white", fontweight="bold")
    for i, p in enumerate(profiles):
        color = PALETTE.get(p.get("model", ""), "#F5F5F5")
        for j in range(len(cols)):
            table[i + 1, j].set_facecolor(color + "33")
    ax.set_title("Model Profil Tablosu", fontsize=13, pad=10)
    plt.tight_layout()
    path = out_dir / "model_profile_table.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Kaydedildi: {path}")


# ── 6. ECE Bar Chart ──────────────────────────────────────────────────────────
def plot_calibration(results: list, out_dir: Path):
    models = [LABELS.get(r["model"], r["model"]) for r in results]
    eces   = [r.get("ece", 0) for r in results]
    colors = [PALETTE.get(r["model"], "gray") for r in results]

    fig, ax = plt.subplots(figsize=(max(7, len(results) * 1.2), 4))
    bars = ax.bar(models, eces, color=colors, alpha=0.85, width=0.6)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    ax.set_ylabel("ECE (dusuk = iyi)")
    ax.set_title("Expected Calibration Error")
    plt.xticks(rotation=20, ha="right")
    ax.set_ylim(0, max(eces) * 1.3 if eces else 0.5)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = out_dir / "calibration_ece.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Kaydedildi: {path}")


# ── 7. Overall CI Karsilastirmasi ─────────────────────────────────────────────
def plot_overall_ci(results: list, out_dir: Path):
    models    = [LABELS.get(r["model"], r["model"]) for r in results]
    baselines = [r["isolation"]["top1_acc"] for r in results]
    cis       = [_overall_ci(r) for r in results]
    colors    = [PALETTE.get(r["model"], "gray") for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Baseline accuracy
    bars1 = axes[0].bar(models, baselines, color=colors, alpha=0.85, width=0.6)
    axes[0].bar_label(bars1, fmt="%.3f", padding=3, fontsize=8)
    axes[0].set_ylabel("Top-1 Accuracy")
    axes[0].set_title("Isolation Baseline Accuracy")
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(axis="y", alpha=0.3)
    plt.setp(axes[0].get_xticklabels(), rotation=20, ha="right")

    # Overall CI
    bars2 = axes[1].bar(models, cis, color=colors, alpha=0.85, width=0.6)
    axes[1].bar_label(bars2, fmt="%.3f", padding=3, fontsize=8)
    axes[1].set_ylabel("Crowding Index (CI)")
    axes[1].set_title("Overall Crowding Index\n(yuksek = crowding'e daha duyarli)")
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(axis="y", alpha=0.3)
    plt.setp(axes[1].get_xticklabels(), rotation=20, ha="right")

    plt.tight_layout()
    path = out_dir / "overall_crowding_index.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Kaydedildi: {path}")


# ── Ana fonksiyon ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="configs/config_openimages.yaml")
    args = parser.parse_args()

    cfg         = load_config(args.config)
    results_dir = Path(cfg["evaluation"]["results_dir"])
    out_dir     = results_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    results  = load_eval_results(results_dir)
    profiles = load_profiles(results_dir)
    categories = get_categories(cfg)

    if not results:
        print("Degerlendirme sonucu bulunamadi. Once evaluate.py calistirin.")
        return

    print(f"\n{len(results)} model sonucu yuklendi. Grafikler uretiliyor...\n")

    plot_spacing_accuracy(results, out_dir)
    plot_crowding_index(results, out_dir)
    plot_per_class_heatmap(results, categories, out_dir)
    plot_confusion_matrices(results, categories, out_dir)
    plot_model_profile_table(profiles, out_dir)
    plot_calibration(results, out_dir)
    plot_overall_ci(results, out_dir)

    print(f"\nTum grafikler kaydedildi: {out_dir}")


if __name__ == "__main__":
    main()
