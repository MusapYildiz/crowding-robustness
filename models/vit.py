"""
vit.py

ViT-S/16 ve ViT-B/16 modellerini timm üzerinden pretrained olarak yükler,
classification head'i num_classes'a göre değiştirir.

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
VIT_VARIANTS = {
    "vit_s_16": "vit_small_patch16_224",   # ~22M
    "vit_b_16": "vit_base_patch16_224",    # ~86M
}


def build_vit(variant: str, num_classes: int,
              pretrained: bool = True,
              freeze_backbone: bool = True) -> nn.Module:
    """
    Args:
        variant         : "vit_s_16" | "vit_b_16"
        num_classes     : çıkış sınıf sayısı
        pretrained      : ImageNet ağırlıklarıyla başla
        freeze_backbone : True → sadece head eğitilir (warm-up)
    """
    if variant not in VIT_VARIANTS:
        raise ValueError(f"Bilinmeyen varyant: {variant}. "
                         f"Seçenekler: {list(VIT_VARIANTS.keys())}")

    timm_name = VIT_VARIANTS[variant]
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


def unfreeze_vit(model: nn.Module, stage: int = 1) -> None:
    """
    ViT backbone'unu kademeli açar.

    stage=1 : son 2 transformer bloğu + head
    stage=2 : son 6 transformer bloğu + head
    stage=3 : tüm model
    """
    if stage == 3:
        for param in model.parameters():
            param.requires_grad = True
        print(f"  ViT unfreeze stage=3: tüm model açıldı")
        return

    # Toplam blok sayısını bul
    n_blocks = len(model.blocks)
    open_from = {1: n_blocks - 2, 2: n_blocks - 6}.get(stage, n_blocks - 2)

    for name, param in model.named_parameters():
        if "head" in name:
            param.requires_grad = True
        elif name.startswith("blocks."):
            block_idx = int(name.split(".")[1])
            if block_idx >= open_from:
                param.requires_grad = True

    opened = sum(p.requires_grad for p in model.parameters())
    print(f"  ViT unfreeze stage={stage}: {opened} parametre aktif "
          f"(blok {open_from}+ açıldı)")


def get_vit_gradcam_layer(model: nn.Module) -> nn.Module:
    """
    Grad-CAM için hedef katman: son transformer bloğunun norm katmanı.
    ViT'te attention map'leri bu katmandan çekilir.
    """
    return model.blocks[-1].norm1


if __name__ == "__main__":
    for variant in VIT_VARIANTS:
        m = build_vit(variant, num_classes=10, pretrained=False)
        n_params = sum(p.numel() for p in m.parameters())
        n_trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"{variant}: toplam={n_params:,}  eğitilebilir={n_trainable:,}")
