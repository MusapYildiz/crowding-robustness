# Crowding Robustness Benchmark

Visual crowding koşullarında ResNet-50, ViT-B/16 ve ConvKAN modellerinin
dayanıklılığını karşılaştıran bir benchmark çalışması.

## Proje Yapısı

```
crowding-robustness/
├── data/
│   ├── prepare_dataset.py   # COCO'dan instance kırpma + split
│   ├── build_composites.py  # Crowding kompozit görsel oluşturma
│   └── dataset.py           # PyTorch Dataset sınıfları
├── models/
│   ├── resnet.py            # ResNet-50
│   ├── vit.py               # ViT-B/16
│   └── convkan.py           # ConvKAN
├── configs/
│   └── config.yaml          # Tüm parametreler
├── notebooks/
│   └── run_colab.ipynb      # Google Colab çalıştırma
├── scripts/
│   └── download_coco.sh     # COCO indirme
├── train.py
├── evaluate.py
├── visualize.py
└── requirements.txt
```

## Kurulum

```bash
pip install -r requirements.txt
```

## Veri Hazırlama

```bash
# 1. COCO indir (Colab'da)
bash scripts/download_coco.sh

# 2. Instance'ları kırp ve split yap
python data/prepare_dataset.py --config configs/config.yaml

# 3. Crowding kompozitlerini oluştur
python data/build_composites.py --config configs/config.yaml
```

## Eğitim

```bash
python train.py --model resnet50 --config configs/config.yaml
python train.py --model vit_b_16 --config configs/config.yaml
python train.py --model convkan  --config configs/config.yaml
```

## Değerlendirme

```bash
python evaluate.py --config configs/config.yaml
```

## Deneysel Tasarım

- **Train**: 10 kategorinin her birinden nesne izolasyonda (uniform background)
- **Test**: Aynı nesneler + sol/sağ flanker, 4 spacing seviyesinde (1°, 2°, 4°, 8°)
- **Flanker tipleri**: same_class / different_class / external
- **Spacing**: Center-to-center, Bouma standardı (1° = 40px)

## Referanslar

- Volokitin et al. (2017). *Do Deep Neural Networks Suffer from Crowding?* NeurIPS.
- Lonnqvist et al. (2020). *Crowding in humans is unlike that in CNNs.*
- Drokin (2024). *Kolmogorov-Arnold Convolutions.* [torch-conv-kan](https://github.com/IvanDrokin/torch-conv-kan)
