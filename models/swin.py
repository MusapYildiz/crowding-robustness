"""
swin.py

Swin-T ve Swin-S modellerini timm üzerinden pretrained olarak yükler,
classification head'i num_classes'a göre değiştirir.

ViT'ten farkı: global self-attention yerine hiyerarşik, kaydırmalı
pencere (shifted window) attention kullanır -- yerel bir alıcı alan
(receptive field) büyüyerek genişler, CNN'in konvolüsyonel hiyerarşisine
ViT'ten daha yakın bir davranış sergiler. Crowding'in yerel özellik
havuzlamasını bozması nedeniyle bu mimari ailesi icin ayrı bir eksen
oluşturur.

timm kullanım gereksinimi:
    pip install timm
"""

import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    raise ImportError("timm kütüphanesi gerekli: pip install timm")


# ── Desteklenen varyantlar ──────────────────────────────────────────────────
SWIN_VARIANTS = {
    "swin_t": "swin_tiny_patch4_window7_224",   # ~28M
    "swin_s": "swin_small_patch4_window7_224",  # ~50M
}


def build_swin(variant: str, num_classes: int,
               pretrained: bool = True,
               freeze_backbone: bool = True) -> nn.Module:
    """
    Args:
        variant         : "swin_t" | "swin_s"
        num_classes     : çıkış sınıf sayısı
        pretrained      : ImageNet ağırlıklarıyla başla
        freeze_backbone : True → sadece head eğitilir (warm-up)
    """
    if variant not in SWIN_VARIANTS:
        raise ValueError(f"Bilinmeyen varyant: {variant}. "
                         f"Seçenekler: {list(SWIN_VARIANTS.keys())}")

    timm_name = SWIN_VARIANTS[variant]
    model = timm.create_model(
        timm_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )

    if freeze_backbone:
        for name, param in model.named_parameters():
            # Sadece head eğitilir
            if "head" not in name:
                param.requires_grad = False

    return model


def unfreeze_swin(model: nn.Module, stage: int = 1) -> None:
    """
    Backbone'u kademeli açar. Swin hiyerarşik oldugu icin (patch merging
    ile küçülen 4 stage), ViT'teki gibi tek düz blok listesi yerine
    stage bazında açılır.

    stage=1 : son stage + head
    stage=2 : son 2 stage + head
    stage=3 : tüm model
    """
    if stage == 3:
        for param in model.parameters():
            param.requires_grad = True
        print(f"  Swin unfreeze stage=3: tüm model açıldı")
        return

    n_stages = len(model.layers)
    open_from = {1: n_stages - 1, 2: n_stages - 2}.get(stage, n_stages - 1)

    for name, param in model.named_parameters():
        if "head" in name:
            param.requires_grad = True
        elif name.startswith("layers."):
            stage_idx = int(name.split(".")[1])
            if stage_idx >= open_from:
                param.requires_grad = True

    opened = sum(p.requires_grad for p in model.parameters())
    print(f"  Swin unfreeze stage={stage}: {opened} parametre aktif "
          f"(stage {open_from}+ açıldı)")


def get_swin_gradcam_layer(model: nn.Module) -> nn.Module:
    """Grad-CAM için hedef katman: son stage'in son bloğunun norm katmanı."""
    return model.layers[-1].blocks[-1].norm1


if __name__ == "__main__":
    for variant in SWIN_VARIANTS:
        m = build_swin(variant, num_classes=10, pretrained=False)
        n_params = sum(p.numel() for p in m.parameters())
        n_trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"{variant}: toplam={n_params:,}  eğitilebilir={n_trainable:,}")
