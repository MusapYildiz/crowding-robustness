"""
convkan.py

ConvKAN-S, ConvKAN-M, ConvKAN-L mimarilerini tanımlar.
torch-conv-kan kütüphanesini kullanır.

Kurulum:
    pip install torch-conv-kan

Her varyant ResNet benzeri bir yapıya sahiptir:
  stem → stage1 → stage2 → stage3 → stage4 → global avg pool → classifier

Parametre hedefleri:
  ConvKAN-S : ~20M
  ConvKAN-M : ~40M
  ConvKAN-L : ~80M
"""

import torch
import torch.nn as nn

try:
    from kan_convs import KANConv2DLayer
except ImportError:
    raise ImportError(
        "torch-conv-kan kütüphanesi gerekli: pip install torch-conv-kan"
    )


# ── Temel yapı taşları ──────────────────────────────────────────────────────

class KANBlock(nn.Module):
    """
    KANConv2D + BatchNorm + KANConv2D + BatchNorm + residual bağlantı.
    ResNet BasicBlock'a benzer ama aktivasyonlar öğrenilebilir spline.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1, grid_size: int = 5):
        super().__init__()

        self.conv1 = KANConv2DLayer(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1,
            grid_size=grid_size,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = KANConv2DLayer(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1,
            grid_size=grid_size,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Boyut uyumsuzluğu varsa projection shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn1(self.conv1(x))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return out


class ConvKAN(nn.Module):
    """
    Genel ConvKAN sınıfı.
    channels ve num_blocks parametreleriyle S/M/L varyantları oluşturulur.
    """

    def __init__(self, channels: list, num_blocks: list,
                 num_classes: int = 10, grid_size: int = 5):
        """
        Args:
            channels   : [stem_ch, stage1_ch, stage2_ch, stage3_ch, stage4_ch]
            num_blocks : [stage1_n, stage2_n, stage3_n, stage4_n]
            num_classes: çıkış sınıf sayısı
            grid_size  : spline grid boyutu (KAN kalitesi vs hız tradeoff)
        """
        super().__init__()
        assert len(channels) == 5 and len(num_blocks) == 4

        # Stem: standart Conv (KAN stem çok yavaş olur)
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7,
                      stride=2, padding=3, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # KAN stage'leri
        self.stage1 = self._make_stage(channels[0], channels[1],
                                       num_blocks[0], stride=1, grid_size=grid_size)
        self.stage2 = self._make_stage(channels[1], channels[2],
                                       num_blocks[1], stride=2, grid_size=grid_size)
        self.stage3 = self._make_stage(channels[2], channels[3],
                                       num_blocks[2], stride=2, grid_size=grid_size)
        self.stage4 = self._make_stage(channels[3], channels[4],
                                       num_blocks[3], stride=2, grid_size=grid_size)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(channels[4], num_classes),
        )

    def _make_stage(self, in_ch: int, out_ch: int, n_blocks: int,
                    stride: int, grid_size: int) -> nn.Sequential:
        layers = [KANBlock(in_ch, out_ch, stride=stride, grid_size=grid_size)]
        for _ in range(1, n_blocks):
            layers.append(KANBlock(out_ch, out_ch, stride=1, grid_size=grid_size))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def get_gradcam_layer(self) -> nn.Module:
        """Grad-CAM için hedef katman: stage4'ün son bloğu."""
        return self.stage4[-1]


# ── Varyant konfigürasyonları ───────────────────────────────────────────────
# channels: [stem, s1, s2, s3, s4]
# num_blocks: [s1, s2, s3, s4]
# Hedef parametre sayıları kalibre edilmiştir.

CONVKAN_CONFIGS = {
    "convkan_s": {
        "channels":   [32, 64,  128, 256, 512],
        "num_blocks": [2,  2,   2,   2],
        "grid_size":  5,
    },
    "convkan_m": {
        "channels":   [64, 128, 256, 512, 512],
        "num_blocks": [2,  3,   4,   2],
        "grid_size":  5,
    },
    "convkan_l": {
        "channels":   [64, 128, 256, 512, 1024],
        "num_blocks": [3,  4,   6,   3],
        "grid_size":  5,
    },
}


def build_convkan(variant: str, num_classes: int,
                  pretrained: bool = False,
                  freeze_backbone: bool = True) -> nn.Module:
    """
    Args:
        variant         : "convkan_s" | "convkan_m" | "convkan_l"
        num_classes     : çıkış sınıf sayısı
        pretrained      : ImageNet pretrained ağırlık yüklenir (varsa)
        freeze_backbone : True → sadece classifier eğitilir (warm-up)

    Not: pretrained=True için torch-conv-kan'ın pretrained ağırlıklarına
         ihtiyaç vardır. Mevcut değilse scratch başlar ve uyarı verir.
    """
    if variant not in CONVKAN_CONFIGS:
        raise ValueError(f"Bilinmeyen varyant: {variant}. "
                         f"Seçenekler: {list(CONVKAN_CONFIGS.keys())}")

    cfg = CONVKAN_CONFIGS[variant]
    model = ConvKAN(
        channels=cfg["channels"],
        num_blocks=cfg["num_blocks"],
        num_classes=num_classes,
        grid_size=cfg["grid_size"],
    )

    if pretrained:
        _load_pretrained(model, variant)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    return model


def _load_pretrained(model: nn.Module, variant: str) -> None:
    """
    torch-conv-kan pretrained ağırlıklarını yükler.
    Mevcut değilse uyarı verir, scratch devam eder.
    """
    try:
        from torch_conv_kan import pretrained_weights
        state = pretrained_weights(variant)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  ConvKAN pretrained yüklendi. "
              f"Missing={len(missing)}, Unexpected={len(unexpected)}")
    except Exception as e:
        print(f"  [WARN] ConvKAN pretrained yüklenemedi ({e}). "
              f"Scratch başlanıyor.")


def unfreeze_convkan(model: nn.Module, stage: int = 1) -> None:
    """
    ConvKAN backbone'unu kademeli açar.

    stage=1 : stage4 + classifier
    stage=2 : stage3 + stage4 + classifier
    stage=3 : tüm model
    """
    if stage == 3:
        for param in model.parameters():
            param.requires_grad = True
        print("  ConvKAN unfreeze stage=3: tüm model açıldı")
        return

    open_layers = {
        1: ["stage4", "classifier"],
        2: ["stage3", "stage4", "classifier"],
    }.get(stage, ["stage4", "classifier"])

    for name, param in model.named_parameters():
        if any(name.startswith(l) for l in open_layers):
            param.requires_grad = True

    opened = sum(p.requires_grad for p in model.parameters())
    print(f"  ConvKAN unfreeze stage={stage}: {opened} parametre aktif")


if __name__ == "__main__":
    for variant in CONVKAN_CONFIGS:
        try:
            m = build_convkan(variant, num_classes=10, pretrained=False,
                              freeze_backbone=False)
            n_params = sum(p.numel() for p in m.parameters())
            n_trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
            print(f"{variant}: toplam={n_params:,}  eğitilebilir={n_trainable:,}")
        except ImportError as e:
            print(f"{variant}: {e}")
