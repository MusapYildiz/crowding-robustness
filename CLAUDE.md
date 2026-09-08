# Crowding Robustness — Proje Özeti

## Proje Nedir?
Görsel crowding etkisinin (flanker nesnelerin hedef nesne tanımayı baskılaması) CNN, Transformer ve KAN mimari ailelerinde nasıl farklılaştığını inceleyen bir benchmark çalışması.

**GitHub repo:** crowding-robustness (Musap'ın hesabında)

## Mevcut Durum

### Tamamlananlar
- Open Images v7 pipeline (30 sınıf, sınıf başına max 500 örnek, SAM2 maske iyileştirme)
- Isolation görseller: 224x224, siyah bg, nesne uzun kenar 80px, aspect ratio korunur
- Crowding kompozitleri: spacing 2°/2.3°/2.5°/2.7°/3° (1°=40px), congruent+incongruent flanker
- **10 model eğitimi tamamlandı** (LP-FT stratejisi): ResNet-34/50/101, ViT-S/16, ViT-B/16, Swin-T, Swin-S, VGG KAGN-11v2/11v4/BN-SA-11v4
- KAN modelleri artık gerçek ImageNet-1K pretrained ağırlıklarla başlıyor (HuggingFace `brivangl/*`) — bkz. Çözülen Sorun
- Swin-T/Swin-S eklendi (windowed attention, `models/swin.py`) ve tüm 10 modelle birlikte değerlendirildi
- Değerlendirme ve görselleştirmeler tamamlandı (10 model için)
- **`crowding_report_v2.md` hazırlandı** — 10 modelin tam sonuçlarıyla (spacing/flanker kırılımı dahil) güncel rapor. Eski `crowding_report.pdf` artık güncelliğini yitirdi (eski KAN scratch sonuçlarını ve Swin'siz 7 modeli içeriyor)

### Sonuçlar (Open Images v7, 30 sınıf, 10 model)
| Model | Iso. Top-1 | CI | ECE |
|---|---|---|---|
| ResNet-34 | 0.7988 | 0.2073 | 0.1423 |
| ResNet-50 | 0.8231 | 0.2533 | 0.2058 |
| ResNet-101 | 0.8347 | 0.2847 | 0.2566 |
| ViT-S/16 | 0.8021 | 0.2583 | 0.2226 |
| ViT-B/16 | 0.7535 | 0.2995 | 0.1367 |
| Swin-T | 0.8557 | 0.1990 | 0.0919 |
| Swin-S | 0.8791 | 0.2323 | 0.1407 |
| VGG KAGN-11v2 | 0.7740 | 0.2709 | 0.0262 |
| VGG KAGN-11v4 | 0.7684 | 0.2331 | 0.0515 |
| VGG KAGN-BN-SA-11v4 | 0.7558 | 0.3366 | 0.0670 |

CI = (Acc_isolation - Acc_crowding) / Acc_isolation

**Gözlenen örüntüler:**
- Kapasite arttıkça CI kötüleşiyor — **dört alt-ailede de tutarlı**: ResNet 34→50→101 (0.207→0.253→0.285), ViT S→B (0.258→0.300), Swin T→S (0.199→0.232), KAN 11v4→11v2→BN-SA (0.233→0.271→0.337)
- **Swin-T tüm 10 model arasında en dayanıklı (CI=0.199) ve isolation'da 2. en isabetli (0.8557)** — yerel pencereli attention, global attention'a (ViT) göre crowding'e belirgin şekilde daha dayanıklı
- ViT-B/16, 10 modelin en düşük isolation accuracy'sine sahip (0.7535) — artık pretrained olan KAN modellerinin bile altında
- En iyi kalibre edilen 3 model KAN ailesinden (ECE 0.026–0.067); Swin-T KAN dışında en iyi kalibre model (0.0919)
- En kırılgan: VGG KAGN-BN-SA-11v4 (CI=0.3366) — en az parametreli KAN olmasına rağmen (12M) self-attention+BN eklenmesi robustness'i düşürmüş
- Detaylı analiz (spacing/flanker kırılımı): `crowding_report_v2.md`

### Çözülen Sorun (önceden: Kritik Sorun)
VGG KAGN-BN ve VGG KAGN modelleri önceden pretrained ağırlıklar yüklenemediğinden scratch başlıyordu (Iso Top-1 ~0.40). `models/convkan.py` düzeltildi: HuggingFace `brivangl/vgg_kagn11_v2`, `vgg_kagn11_v4`, `vgg_kagn_bn11sa_v4` repo'larının `config.json`'larından birebir kopyalanan mimari kwargs'larla model kuruluyor, `model.safetensors` yükleniyor, `load_state_dict` tam eşleşmezse (missing/unexpected key) artık sessizce scratch'e düşmek yerine hata fırlatıyor. Sonuç: Iso Top-1 ~0.40 → ~0.75-0.77, CNN/ViT ile adil karşılaştırma artık mümkün.

### Çözülen Görev: Swin Transformer Ekleme
Crowding, hedefin etrafındaki yerel özellik havuzlamasının flanker'lar tarafından bozulmasıyla ilgili. ViT global attention kullanıyor; Swin ise hiyerarşik + kaydırmalı pencere (shifted window) attention ile CNN'in yerel/hiyerarşik yapısına daha yakın. `models/swin.py` eklendi (timm: `swin_tiny_patch4_window7_224`/`swin_t` ~27.5M, `swin_small_patch4_window7_224`/`swin_s` ~48.9M), `train.py`/`evaluate.py`/`configs/config_openimages.yaml`/`visualize.py`/notebook güncellendi, Colab'da eğitilip değerlendirildi. Sonuç yukarıdaki tabloda.

## Sıradaki Görev: Beyaz Arka Plan — Ek Deneysel Kol (aktif)

**Motivasyon:** Siyah arka planla eğitilmiş ResNet-34'ün, sadece arka planı beyaza çevrilmiş test görsellerinde sıfır-shot değerlendirilmesiyle yapılan hızlı bir kontrol testi, isolation accuracy'de **%21 relatif düşüş** (0.7988→0.6321) gösterdi — modelin siyah arka plan istatistiğine bağımlı kaldığını, saf nesne tanıma özelliklerinden ayrıştıramadığını gösteriyor. (Not: bu testte CI görünürde iyileşti ama bu bir floor-effect — hem isolation hem crowding accuracy düştü, isolation orantısız daha çok düştüğü için CI küçüldü; "daha dayanıklı" diye yorumlanmamalı.)

**Karar:** Beyaz arka planı **siyahın yerine değil, ek bir deneysel kol olarak** tam pipeline'dan geçirip 10 modeli sıfırdan eğitmek. Siyah pipeline/notebook/sonuçlar hiç değiştirilmedi.

**Durum: kod tarafı hazır**, Colab'da çalıştırılmayı bekliyor.
- **Bug fix (kritik):** `build_composites.py`'deki `place_flanker_on_canvas`, flanker maskesini `pixel > bg_color + 10` ile ayırıyordu — bu, arka planın nesnelerden koyu olduğunu varsayıyor. `background_color=255` ile bu koşul hiç sağlanmaz, flanker'lar tamamen görünmez olurdu. Düzeltme: `abs(pixel - bg_color) > 10` (yön-bağımsız, hem siyah hem beyaz için çalışır).
- **`configs/config_openimages_white.yaml`** eklendi — `config_openimages.yaml` ile birebir aynı (30 sınıf, spacing, LP-FT hiperparametreleri, 10 model listesi), tek fark: `background_color: 255`, `output_dir: ./dataset_openimages_white`, `results_dir: ./results_openimages_white`.
- **`notebooks/run_openimages_white.ipynb`** eklendi — `run_openimages.ipynb`'nin birebir kopyası, sadece config yolu ve Drive hedef klasörleri (`dataset_openimages_white/`, `results_openimages_white/`) güncellendi. Siyah notebook değişmedi.
- **Gerekli tam pipeline (fiftyone raw download + SAM2 dahil):** Siyah veri setinin ham görüntüleri/SAM2 maskeleri Drive'a kaydedilmedi (sadece nihai `dataset_openimages/` composite'leri var) — bu yüzden beyaz için `download.py` ve `refine_masks.py`'nin de sıfırdan çalıştırılması gerekiyor, sadece `prepare_dataset.py`/`build_composites.py` değil.
- Sıradaki adım: `run_openimages_white.ipynb`'yi Colab'da baştan sona çalıştırmak (indirme → SAM2 → isolation → composite → 10 model LP-FT → evaluate → visualize → Drive'a kaydet), sonra `crowding_report_v2.md`'ye siyah-vs-beyaz karşılaştırma bölümü eklemek.

### Diğer aday konular (ertelendi)
- Grad-CAM analizi (her `models/*.py`'de `get_*_gradcam_layer` fonksiyonları hazır ama hiç kullanılmıyor)
- İnsan psikofiziği karşılaştırması
- Dikey eksen crowding (eksantriklik varyasyonu)
- KAN modelleri için FLOPs hesaplama (fvcore custom KAN op'larını desteklemiyor)
- External flanker koşulu **teknik olarak zaten üretiliyor** (`build_composites.py`'de `get_external_pool` — hedef 30 sınıf dışındaki Open Images görselleri kullanıyor) ve `evaluation_summary.json`'da her model için mevcut, ama bu görseller hâlâ aynı Open Images dağılımından geldiği için gerçek anlamda "dış-domain" değil — bu yüzden bilinçli olarak rapordan çıkarıldı. Gerçek bir external flanker seti için farklı bir veri kaynağından (Open Images dışı) görsel gerekir.

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
│   ├── config_openimages.yaml       ← siyah arka plan (aktif, tamamlandı)
│   └── config_openimages_white.yaml ← beyaz arka plan (YENİ, eğitim bekliyor)
├── data/
│   └── openimages/
│       ├── download.py           ← fiftyone ile dengeli indirme
│       ├── refine_masks.py       ← SAM2 maske iyileştirme
│       ├── prepare_dataset.py    ← isolation görseller
│       ├── build_composites.py   ← crowding kompozitleri (bg-renk-bağımsız maskeleme düzeltildi)
│       └── dataset.py            ← PyTorch Dataset
├── models/
│   ├── resnet.py
│   ├── vit.py
│   ├── swin.py                   ← Swin-T/S (eğitim tamamlandı)
│   └── convkan.py                ← VGG KAGN modelleri (pretrained düzeltildi)
├── train.py                      ← LP-FT eğitim
├── evaluate.py                   ← CI, ECE, spacing curve
├── visualize.py                  ← 7 grafik türü
└── notebooks/
    ├── run_openimages.ipynb        ← Colab notebook (siyah, tamamlandı)
    └── run_openimages_white.ipynb  ← Colab notebook (beyaz, YENİ)
```

## Drive Konumları
- `/content/drive/MyDrive/crowding_openimages/` — ana proje
- `/content/drive/MyDrive/crowding_openimages/results_openimages/` — sonuçlar (siyah)
- `/content/drive/MyDrive/crowding_openimages/dataset_openimages/` — veri (siyah)
- `/content/drive/MyDrive/crowding_openimages/results_openimages_white/` — sonuçlar (beyaz, YENİ)
- `/content/drive/MyDrive/crowding_openimages/dataset_openimages_white/` — veri (beyaz, YENİ)

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
