"""
models/convkan.py

VGG KAGN BN 11v4 ve VGG KAGN 11v4 modellerini
torch-conv-kan reposundan yukler.

Pretrained ImageNet-1K agirliklari HuggingFace'de mevcut:
  huggingface.co/brivangl

Kaynak: Drokin, I. "Kolmogorov-Arnold Convolutions:
        Design Principles and Empirical Studies"
        arXiv:2407.01092 (2024)

Kurulum:
    git clone https://github.com/IvanDrokin/torch-conv-kan.git
    pip install -r torch-conv-kan/requirements.txt
    # sys.path'e ekle veya pip install -e torch-conv-kan
"""

import torch
import torch.nn as nn


CONVKAN_VARIANTS = {
    "vgg_kagn_bn_11v4": {
        "model_name": "vgg_kagn_bn_11v4",
        "hf_repo":    "brivangl/vgg_kagn_bn_11v4",
        "params":     "7.25M",
        "top1":       68.5,
    },
    "vgg_kagn_11v4": {
        "model_name": "vgg_kagn_11v4",
        "hf_repo":    "brivangl/vgg_kagn_11v4",
        "params":     "~15M",
        "top1":       61.2,
    },
}


def _load_from_torch_conv_kan(model_name: str, num_classes: int,
                                pretrained: bool) -> nn.Module:
    """
    torch-conv-kan reposundan model yukler.
    Pretrained=True ise HuggingFace'den agirliklar indirilir.
    """
    # torch-conv-kan'i dogrudan import et
    # NOT: torch-conv-kan sys.path'e EKLENMEMELI
    # cunku kendi 'models/' dizini bizimkiyle catisiyor.
    # Bunun yerine importlib ile direkt dosyadan yukluyoruz.
    import importlib.util, sys, os

    def _load_from_file(module_name, file_path):
        spec = importlib.util.spec_from_file_location(
            module_name, file_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        return mod

    # torch-conv-kan konumunu bul
    possible_paths = [
        "/content/crowding-robustness/torch-conv-kan",
        "./torch-conv-kan",
        os.path.join(os.path.dirname(__file__), "../torch-conv-kan"),
    ]
    tck_root = None
    for p in possible_paths:
        if os.path.exists(p):
            tck_root = os.path.abspath(p)
            break

    if tck_root is None:
        raise ImportError(
            "torch-conv-kan bulunamadi.\n"
            "git clone https://github.com/IvanDrokin/torch-conv-kan.git"
        )

    # Gerekli bagimliliklari yukle
    for dep in ["kan_convs", "kans"]:
        dep_path = os.path.join(tck_root, dep)
        if os.path.exists(dep_path + ".py"):
            _load_from_file(dep, dep_path + ".py")
        elif os.path.exists(dep_path):
            # paket
            init = os.path.join(dep_path, "__init__.py")
            if os.path.exists(init):
                _load_from_file(dep, init)

    vgg_kan_path = os.path.join(tck_root, "models", "vgg_kan.py")
    if not os.path.exists(vgg_kan_path):
        raise ImportError(f"vgg_kan.py bulunamadi: {vgg_kan_path}")

    vgg_kan_mod = _load_from_file("_vgg_kan_internal", vgg_kan_path)
    vgg11_kan    = vgg_kan_mod.vgg11_kan
    vgg11_kan_bn = vgg_kan_mod.vgg11_kan_bn

    if model_name == "vgg_kagn_bn_11v4":
        model = vgg11_kan_bn(num_classes=1000)
    elif model_name == "vgg_kagn_11v4":
        model = vgg11_kan(num_classes=1000)
    else:
        raise ValueError(f"Bilinmeyen variant: {model_name}")

    if pretrained:
        _load_pretrained_weights(model, model_name)

    # Classifier katmanini degistir
    model = _replace_classifier(model, num_classes)

    return model


def _load_pretrained_weights(model: nn.Module, model_name: str):
    """HuggingFace'den pretrained agirliklari yukler."""
    try:
        from huggingface_hub import hf_hub_download
        import os

        repo_id   = CONVKAN_VARIANTS[model_name]["hf_repo"]
        ckpt_path = hf_hub_download(repo_id=repo_id,
                                     filename="model.pth")
        state = torch.load(ckpt_path, map_location="cpu")

        # State dict key'leri farkli olabilir
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model" in state:
            state = state["model"]

        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  [{model_name}] Pretrained yuklendi. "
              f"Missing={len(missing)}, Unexpected={len(unexpected)}")

    except Exception as e:
        print(f"  [WARN] {model_name} pretrained yuklenemedi: {e}")
        print("  Scratch baslanıyor.")


def _replace_classifier(model: nn.Module,
                         num_classes: int) -> nn.Module:
    """
    Son classifier katmanini num_classes'a gore degistirir.
    torch-conv-kan modellerinde classifier yapisi farkli olabilir.
    """
    # VGG tarzı modellerde genellikle model.classifier[-1] veya model.fc
    replaced = False

    if hasattr(model, "classifier"):
        clf = model.classifier
        if isinstance(clf, nn.Sequential):
            # Son Linear katmani bul ve degistir
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
        in_features  = model.fc.in_features
        model.fc     = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
        replaced = True

    if not replaced:
        print(f"  [WARN] Classifier katmani bulunamadi, "
              f"model degistirilmedi.")

    return model


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

    model = _load_from_torch_conv_kan(variant, num_classes, pretrained)

    if freeze_backbone:
        for name, param in model.named_parameters():
            # Classifier disindaki her seyi dondur
            if "classifier" not in name and "fc" not in name:
                param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    print(f"  {variant}: {trainable:,} egitilebilir parametre")

    return model


def unfreeze_convkan(model: nn.Module, stage: int = 1) -> None:
    """
    LP-FT: stage=1 -> tum model aciilr (Full Fine-Tuning)
    """
    for param in model.parameters():
        param.requires_grad = True

    total = sum(p.numel() for p in model.parameters())
    print(f"  ConvKAN unfreeze (stage={stage}): "
          f"{total:,} parametre aktif")


def get_convkan_gradcam_layer(model: nn.Module) -> nn.Module:
    """Grad-CAM icin hedef katman: features'in son konv katmani."""
    if hasattr(model, "features"):
        # Son Conv2d katmanini bul
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
