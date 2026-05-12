"""
models/convkan.py

VGG KAGN BN 11v4 ve VGG KAGN 11v4 modellerini
torch-conv-kan reposundan yukler.

Pretrained ImageNet-1K agirliklari HuggingFace'de mevcut:
  huggingface.co/brivangl
"""

import os, sys, importlib, importlib.util, types
import torch
import torch.nn as nn


CONVKAN_VARIANTS = {
    "vgg_kagn_bn_11v4": {
        "hf_repo": "brivangl/vgg_kagn_bn_11v4",
        "builder": "vggkagn_bn",
    },
    "vgg_kagn_11v4": {
        "hf_repo": "brivangl/vgg_kagn_11v4",
        "builder": "vggkagn",
    },
}


def _find_tck_root() -> str:
    """torch-conv-kan repo dizinini bulur."""
    candidates = [
        "/content/crowding-robustness/torch-conv-kan",
        os.path.join(os.path.dirname(__file__), "../torch-conv-kan"),
        "./torch-conv-kan",
    ]
    for p in candidates:
        if os.path.exists(p):
            return os.path.abspath(p)
    raise ImportError(
        "torch-conv-kan bulunamadi.\n"
        "git clone https://github.com/IvanDrokin/torch-conv-kan.git"
    )


def _load_tck_modules(tck_root: str):
    """
    torch-conv-kan modullerini sys.path catismasi olmadan yukler.
    
    Yontem:
      1. tck_root'u gecici olarak sys.path'e ekle
      2. kans paketini tamamen yukle (submoduller dahil)
      3. kan_convs ve models/vgg_kan'i yukle
      4. tck_root'u sys.path'den cikar
      (yuklenmis moduller sys.modules'de kalir)
    """
    # Eski eksik yuklemeleri temizle
    for key in list(sys.modules.keys()):
        if key in ('kans', 'kan_convs') or \
           key.startswith('kans.') or \
           key.startswith('kan_convs.'):
            del sys.modules[key]

    # Gecici path ekle
    if tck_root not in sys.path:
        sys.path.insert(0, tck_root)

    try:
        # kans paketini tam yukle
        import kans
        if not hasattr(kans, 'RadialBasisFunction'):
            raise ImportError("kans.RadialBasisFunction bulunamadi")

        # kan_convs yukle
        import kan_convs

        # models/vgg_kan yukle
        vgg_kan_path = os.path.join(tck_root, "models", "vggkan.py")
        if not os.path.exists(vgg_kan_path):
            raise ImportError(f"vgg_kan.py bulunamadi: {vgg_kan_path}")

        spec = importlib.util.spec_from_file_location(
            "_tck_vgg_kan", vgg_kan_path
        )
        vgg_kan_mod = importlib.util.module_from_spec(spec)
        sys.modules["_tck_vgg_kan"] = vgg_kan_mod
        spec.loader.exec_module(vgg_kan_mod)

        return vgg_kan_mod

    finally:
        # tck_root'u path'den cikar (models/ catismasini onle)
        if tck_root in sys.path:
            sys.path.remove(tck_root)


def _replace_classifier(model: nn.Module, num_classes: int) -> nn.Module:
    """Son classifier katmanini num_classes'a gore degistirir."""
    replaced = False

    if hasattr(model, "classifier"):
        clf = model.classifier
        if isinstance(clf, nn.Sequential):
            for i in range(len(clf) - 1, -1, -1):
                if isinstance(clf[i], nn.Linear):
                    in_features = clf[i].in_features
                    clf[i] = nn.Sequential(
                        nn.Dropout(p=0.3),
                        nn.Linear(in_features, num_classes)
                    )
                    replaced = True
                    break
        elif isinstance(clf, nn.Linear):
            in_features = clf.in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_features, num_classes)
            )
            replaced = True

    if not replaced and hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
        replaced = True

    if not replaced:
        print(f"  [WARN] Classifier katmani bulunamadi.")

    return model


def _load_pretrained(model: nn.Module, variant: str):
    """HuggingFace'den pretrained agirliklari yukler."""
    try:
        from huggingface_hub import hf_hub_download
        repo_id   = CONVKAN_VARIANTS[variant]["hf_repo"]
        ckpt_path = hf_hub_download(repo_id=repo_id, filename="model.pth")
        state = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in state: state = state["state_dict"]
        elif "model" in state:    state = state["model"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  [{variant}] Pretrained yuklendi. "
              f"Missing={len(missing)}, Unexpected={len(unexpected)}")
    except Exception as e:
        print(f"  [WARN] {variant} pretrained yuklenemedi: {e}")
        print("  Scratch baslanıyor.")


def build_convkan(variant: str, num_classes: int,
                  pretrained: bool = True,
                  freeze_backbone: bool = True) -> nn.Module:
    """
    Args:
        variant         : "vgg_kagn_bn_11v4" | "vgg_kagn_11v4"
        num_classes     : cikis sinif sayisi
        pretrained      : HuggingFace'den ImageNet agirliklari yukle
        freeze_backbone : True -> sadece classifier egitilir (LP asama)
    """
    if variant not in CONVKAN_VARIANTS:
        raise ValueError(
            f"Bilinmeyen variant: {variant}. "
            f"Secenekler: {list(CONVKAN_VARIANTS.keys())}"
        )

    tck_root    = _find_tck_root()
    vgg_kan_mod = _load_tck_modules(tck_root)

    builder_name = CONVKAN_VARIANTS[variant]["builder"]
    builder      = getattr(vgg_kan_mod, builder_name)
    # vggkan.py fonksiyonlari: (input_channels, num_classes, ...)
    model        = builder(input_channels=3, num_classes=1000)

    if pretrained:
        _load_pretrained(model, variant)

    model = _replace_classifier(model, num_classes)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if "classifier" not in name and "fc" not in name:
                param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    print(f"  {variant}: {trainable:,} egitilebilir parametre")

    return model


def unfreeze_convkan(model: nn.Module, stage: int = 1) -> None:
    """LP-FT: tum modeli ac."""
    for param in model.parameters():
        param.requires_grad = True
    total = sum(p.numel() for p in model.parameters())
    print(f"  ConvKAN unfreeze: {total:,} parametre aktif")


def get_convkan_gradcam_layer(model: nn.Module) -> nn.Module:
    """Grad-CAM icin hedef katman: son Conv2d."""
    if hasattr(model, "features"):
        last_conv = None
        for m in model.features.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        if last_conv:
            return last_conv
    return list(model.modules())[-2]


if __name__ == "__main__":
    for variant in CONVKAN_VARIANTS:
        try:
            m = build_convkan(variant, num_classes=30,
                              pretrained=False, freeze_backbone=False)
            n = sum(p.numel() for p in m.parameters())
            print(f"{variant}: {n:,} toplam parametre")
        except Exception as e:
            print(f"{variant}: {e}")
