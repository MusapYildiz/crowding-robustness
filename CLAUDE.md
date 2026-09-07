# Crowding Robustness — Proje Özeti

## Proje Nedir?
Görsel crowding etkisinin (flanker nesnelerin hedef nesne tanımayı baskılaması) CNN, Transformer ve KAN mimari ailelerinde nasıl farklılaştığını inceleyen bir benchmark çalışması.

**GitHub repo:** crowding-robustness (Musap'ın hesabında)

## Mevcut Durum

### Tamamlananlar
- Open Images v7 pipeline (30 sınıf, sınıf başına max 500 örnek, SAM2 maske iyileştirme)
- Isolation görseller: 224x224, siyah bg, nesne uzun kenar 80px, aspect ratio korunur
- Crowding kompozitleri: spacing 2°/2.3°/2.5°/2.7°/3° (1°=40px), congruent+incongruent flanker
- 8 model eğitimi tamamlandı (LP-FT stratejisi): ResNet-34/50/101, ViT-S/16, ViT-B/16, VGG KAGN-11v2/11v4/BN-SA-11v4
- KAN modelleri artık gerçek ImageNet-1K pretrained ağırlıklarla başlıyor (HuggingFace `brivangl/*`) — bkz. Çözülen Sorun
- Değerlendirme ve görselleştirmeler tamamlandı (8 model için)
- Word raporu hazırlandı (KAN pretrained sonuçlarıyla güncellenmesi gerekiyor)

### Sonuçlar (Open Images v7, 30 sınıf)
| Model | Iso. Top-1 | CI | ECE |
|---|---|---|---|
| ResNet-34 | 0.7988 | 0.2073 | 0.1423 |
| ResNet-50 | 0.8231 | 0.2533 | 0.2058 |
| ResNet-101 | 0.8347 | 0.2847 | 0.2566 |
| ViT-S/16 | 0.8021 | 0.2583 | 0.2226 |
| ViT-B/16 | 0.7535 | 0.2995 | 0.1367 |
| VGG KAGN-11v2 | 0.7740 | 0.2709 | 0.0262 |
| VGG KAGN-11v4 | 0.7684 | 0.2331 | 0.0515 |
| VGG KAGN-BN-SA-11v4 | 0.7558 | 0.3366 | 0.0670 |

CI = (Acc_isolation - Acc_crowding) / Acc_isolation

**Gözlenen örüntüler:**
- Kapasite arttıkça CI kötüleşiyor — üç ailede de tutarlı: ResNet 34→50→101 (0.207→0.253→0.285), ViT S→B (0.258→0.300), KAN 11v4→11v2→BN-SA (0.233→0.271→0.337)
- KAN ailesi belirgin şekilde daha iyi kalibre (ECE 0.026–0.067) — CNN/ViT'in 0.14–0.26 aralığının çok altında
- En dayanıklı (en düşük CI): ResNet-34. En kırılgan: VGG KAGN-BN-SA-11v4

### Çözülen Sorun (önceden: Kritik Sorun)
VGG KAGN-BN ve VGG KAGN modelleri önceden pretrained ağırlıklar yüklenemediğinden scratch başlıyordu (Iso Top-1 ~0.40). `models/convkan.py` düzeltildi: HuggingFace `brivangl/vgg_kagn11_v2`, `vgg_kagn11_v4`, `vgg_kagn_bn11sa_v4` repo'larının `config.json`'larından birebir kopyalanan mimari kwargs'larla model kuruluyor, `model.safetensors` yükleniyor, `load_state_dict` tam eşleşmezse (missing/unexpected key) artık sessizce scratch'e düşmek yerine hata fırlatıyor. Sonuç: Iso Top-1 ~0.40 → ~0.75-0.77, CNN/ViT ile adil karşılaştırma artık mümkün.

## Sıradaki Görev: Swin Transformer Ekleme

Crowding, hedefin etrafındaki yerel özellik havuzlamasının flanker'lar tarafından bozulmasıyla ilgili. ViT global attention kullanıyor; Swin ise hiyerarşik + kaydırmalı pencere (shifted window) attention ile CNN'in yerel/hiyerarşik yapısına daha yakın — bu nedenle Transformer ailesi içinde "yerel vs. global attention" eksenini test etmek için eklendi.

**Durum: kod tarafı tamamlandı**, eğitim/değerlendirme Colab'da yapılmayı bekliyor.
- `models/swin.py` eklendi — timm üzerinden `swin_tiny_patch4_window7_224` (`swin_t`, ~28M) ve `swin_small_patch4_window7_224` (`swin_s`, ~50M), ViT-S/ViT-B'ye benzer boyut karşılaştırması için
- `train.py`, `evaluate.py`, `configs/config_openimages.yaml`, `visualize.py`, `notebooks/run_openimages.ipynb` güncellendi
- Sıradaki adım: Colab'da `swin_t` ve `swin_s` için LP-FT eğitimi + tüm modellerle birlikte değerlendirme (bkz. 3 KAN modeli için izlenen Drive tabanlı iş akışı — aynı desen uygulanabilir)

## Mimari Detaylar

### Flanker Yapısı
```
[FLANKER] [HEDEF] [FLANKER]
Sol ve sağ: aynı instance
Her spacing seviyesinde: aynı flanker (sabit)
Canvas: 224x224 siyah
```

### LP-FT Eğitim Stratejisi
- LP: 15 epoch, backbone frozen, lr=0.001
- FT: 30 epoch, tüm model, lr=0.0001→0.000001 cosine decay
- AdamW, weight_decay=0.0001, grad_clip=1.0, label_smoothing=0.1

### Kritik Path Sorunu (Colab'da)
torch-conv-kan ve crowding-robustness'ın ikisinde de `models/` dizini var — çakışma yaşanıyor.
**Çözüm:** torch-conv-kan asla sys.path'e eklenmemeli, importlib ile yüklenmeli.

```python
# Her Colab session başında:
sys.path = [p for p in sys.path
            if 'torch-conv-kan' not in p
            and '/sam2' not in p
            and p != '/content']
sys.path.insert(0, '/content/crowding-robustness')
```

### SAM2 Kurulumu (Colab)
```bash
git clone https://github.com/facebookresearch/sam2.git /content/sam2
cd /content/sam2 && pip install -e .
wget -P /content/sam2_weights \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
```
SAM2 da sys.path'e eklenmemeli — site-packages'tan import edilmeli.

## Dosya Yapısı
```
crowding-robustness/
├── configs/
│   └── config_openimages.yaml    ← aktif config
├── data/
│   └── openimages/
│       ├── download.py           ← fiftyone ile dengeli indirme
│       ├── refine_masks.py       ← SAM2 maske iyileştirme
│       ├── prepare_dataset.py    ← isolation görseller
│       ├── build_composites.py   ← crowding kompozitleri
│       └── dataset.py            ← PyTorch Dataset
├── models/
│   ├── resnet.py
│   ├── vit.py
│   ├── swin.py                   ← Swin-T/S (YENİ, eğitim bekliyor)
│   └── convkan.py                ← VGG KAGN modelleri (pretrained düzeltildi)
├── train.py                      ← LP-FT eğitim
├── evaluate.py                   ← CI, ECE, spacing curve
├── visualize.py                  ← 7 grafik türü
└── notebooks/
    └── run_openimages.ipynb      ← Colab notebook
```

## Drive Konumları
- `/content/drive/MyDrive/crowding_openimages/` — ana proje
- `/content/drive/MyDrive/crowding_openimages/results_openimages/` — sonuçlar
- `/content/drive/MyDrive/crowding_openimages/dataset_openimages/` — veri

## Sınıflar (30 adet)
Car, Person, Bottle, Dog, Airplane, Bird, Motorcycle, Horse, Cat, Book,
Truck, Bus, Laptop, Cake, Elephant, Backpack, Sheep, Couch, Pizza, Toilet,
Clock, Giraffe, Zebra, Bear, Tiger, Traffic light, Stop sign, Fire hydrant,
Skateboard, Suitcase

## ImageNet-1K Pretrained Baseline Doğrulukları
| Model | Top-1 | Top-5 |
|---|---|---|
| ResNet-34 | %73.3 | %91.4 |
| ResNet-50 | %80.9 | %95.4 |
| ResNet-101 | %81.9 | %95.8 |
| ViT-S/16 | %78.3 | %94.1 |
| ViT-B/16 | %81.1 | %95.6 |
| Swin-T | %81.2 | %95.5 |
| Swin-S | %83.0 | %96.2 |
| vgg_kagn11_v2 | %59.1 | %82.3 |
| vgg_kagn11_v4 | — | — |
| vgg_kagn_bn11sa_v4 | %68.5 | — |
