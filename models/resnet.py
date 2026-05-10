"""
resnet.py

ResNet-34, ResNet-50, ResNet-101 modellerini pretrained olarak yükler,
classifier katmanını num_classes'a göre değiştirir.

Eğitim iki aşamalı:
  1. Warm-up : sadece classifier katmanı eğitilir (backbone dondurulur)
  2. Fine-tune: backbone kademeli olarak açılır
"""

import torch
import torch.nn as nn
from torchvision import models


# ── Desteklenen varyantlar ──────────────────────────────────────────────────
RESNET_VARIANTS = {
    "resnet34":  (models.resnet34,  models.ResNet34_Weights.IMAGENET1K_V1),
    "resnet50":  (models.resnet50,  models.ResNet50_Weights.IMAGENET1K_V1),
    "resnet101": (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V1),
}


def build_resnet(variant: str, num_classes: int,
                 pretrained: bool = True,
                 freeze_backbone: bool = True) -> nn.Module:
    """
    Args:
        variant         : "resnet34" | "resnet50" | "resnet101"
        num_classes     : çıkış sınıf sayısı
        pretrained      : ImageNet ağırlıklarıyla başla
        freeze_backbone : True → sadece classifier eğitilir (warm-up)
    """
    if variant not in RESNET_VARIANTS:
        raise ValueError(f"Bilinmeyen varyant: {variant}. "
                         f"Seçenekler: {list(RESNET_VARIANTS.keys())}")

    model_fn, weights = RESNET_VARIANTS[variant]
    model = model_fn(weights=weights if pretrained else None)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Classifier'ı değiştir
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )

    return model


def unfreeze_resnet(model: nn.Module, stage: int = 1) -> None:
    """
    Backbone'u kademeli açar.

    stage=1 : layer4 + fc
    stage=2 : layer3 + layer4 + fc
    stage=3 : tüm model
    """
    layer_map = {
        1: ["layer4", "fc"],
        2: ["layer3", "layer4", "fc"],
        3: ["layer1", "layer2", "layer3", "layer4", "fc"],
    }
    target_layers = layer_map.get(stage, ["layer4", "fc"])

    for name, param in model.named_parameters():
        if any(name.startswith(l) for l in target_layers):
            param.requires_grad = True

    opened = sum(p.requires_grad for p in model.parameters())
    print(f"  ResNet unfreeze stage={stage}: {opened} parametre aktif")


def get_resnet_gradcam_layer(model: nn.Module) -> nn.Module:
    """Grad-CAM için hedef katman: layer4'ün son bloğu."""
    return model.layer4[-1]


if __name__ == "__main__":
    for variant in RESNET_VARIANTS:
        m = build_resnet(variant, num_classes=10, pretrained=False)
        n_params = sum(p.numel() for p in m.parameters())
        n_trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"{variant}: toplam={n_params:,}  eğitilebilir={n_trainable:,}")
