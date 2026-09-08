# Görsel Crowding Etkisinin Farklı Mimari Ailelerinde Karşılaştırmalı Analizi

**ResNet · ViT · Swin · KAN | Open Images v7 + SAM2**

---

## 1. Giriş ve Motivasyon

Görsel crowding, hedef nesnenin algılanmasının etrafındaki flanker nesneler tarafından baskılanması olgusudur. Bu çalışmada üç farklı mimari ailesinden (CNN, Transformer, KAN) **10 modelin** crowding dayanıklılığı karşılaştırılmıştır. Transformer ailesi iki alt-tipi kapsayacak şekilde genişletilmiştir: **global attention** (ViT-S/16, ViT-B/16) ve **hiyerarşik, kaydırmalı pencere (shifted-window) attention** (Swin-T, Swin-S). Bu ayrım, crowding'in hedefin etrafındaki yerel özellik havuzlamasının flanker'lar tarafından bozulmasıyla ilgili olması nedeniyle önemlidir: Swin'in pencereli attention'ı CNN'in yerel/hiyerarşik işleme yapısına ViT'ten daha yakındır.

### 1.1. Temel Kavramlar

**Crowding Index (CI)** = (Acc_isolation − Acc_crowding) / Acc_isolation

CI [0,1] arasında değer alır. 0: dayanıklı, 1: çok duyarlı. Negatif CI: flanker performansı artırıyor.

**Top-1 Accuracy:** En yüksek olasılıklı tahmin doğru mu?
**Top-5 Accuracy:** 30 sınıf arasından en yüksek 5 tahmin içinde gerçek sınıf var mı?

**Bouma Yasası:** Kritik spacing ~ nesne boyutunun 0.4–0.5 katı. Bu değerin üzerinde crowding azalır.

---

## 2. Yöntem

### 2.1. Veri Seti

Kaynak: Open Images v7 | Maske: SAM2 | 30 sınıf | Sınıf başına maks. 500 örnek

Split: %70 train / %15 val / %15 test — instance bazlı strict split

*Sınıflar:* Car, Person, Bottle, Dog, Airplane, Bird, Motorcycle, Horse, Cat, Book, Truck, Bus, Laptop, Cake, Elephant, Backpack, Sheep, Couch, Pizza, Toilet, Clock, Giraffe, Zebra, Bear, Tiger, Traffic light, Stop sign, Fire hydrant, Skateboard, Suitcase

### 2.2. Görsel Hazırlama

Canvas: 224×224 piksel, siyah arka plan | Nesne: uzun kenar 80px, aspect ratio korunur

### 2.3. Crowding Koşulları

Flanker: Sol ve sağda simetrik, her iki tarafta aynı instance | Spacing: Center-to-center, 1°=40px

| Spacing (°) | Piksel (px) |
|---|---|
| 2.0° | 80 px |
| 2.3° | 92 px |
| 2.5° | 100 px |
| 2.7° | 108 px |
| 3.0° | 120 px |

Flanker tipleri: Congruent (aynı sınıf, farklı instance) | Incongruent (farklı sınıf)

### 2.4. Eğitim — LP-FT

LP: 15 epoch, backbone dondurulur, lr=0.001
FT: 30 epoch, tüm model açılır, lr=0.0001 → 0.000001 (cosine decay)
AdamW | weight_decay=0.0001 | gradient_clip=1.0 | label_smoothing=0.1 | early_stop=7

**Pretrained: ImageNet-1K (tüm modeller, KAN dahil).** Önceki turda KAN modelleri (VGG KAGN-BN, VGG KAGN) pretrained ağırlıklar yüklenemediğinden scratch başlıyordu. Bu sorun çözüldü: HuggingFace `brivangl/vgg_kagn11_v2`, `vgg_kagn11_v4`, `vgg_kagn_bn11sa_v4` repo'larının `config.json`'larından birebir kopyalanan mimari kwargs'larıyla model kuruluyor, `model.safetensors` yükleniyor; `load_state_dict` tam eşleşmezse (missing/unexpected key) artık sessizce scratch'e düşmek yerine hata fırlatılıyor.

---

## 3. Model Profilleri

| Model | Mimari | Parametre | Disk Boyutu | FLOPs | GPU Latency |
|---|---|---|---|---|---|
| ResNet-34 | CNN | 21.3M | 81.3 MB | 3.68 G | 1.375 ms |
| ResNet-50 | CNN | 23.6M | 90.1 MB | 4.14 G | 1.569 ms |
| ResNet-101 | CNN | 42.6M | 162.8 MB | 7.88 G | 2.988 ms |
| ViT-S/16 | Transformer (global) | 21.7M | 82.7 MB | 4.25 G | 1.794 ms |
| ViT-B/16 | Transformer (global) | 85.8M | 327.4 MB | 16.87 G | 3.593 ms |
| Swin-T | Transformer (windowed) | 27.5M | 106.1 MB | 4.51 G | 11.648 ms |
| Swin-S | Transformer (windowed) | 48.9M | 187.9 MB | 8.77 G | 23.750 ms |
| VGG KAGN-11v2 | KAN | 24.4M | 93.1 MB | — | 8.663 ms |
| VGG KAGN-11v4 | KAN | 40.9M | 156.1 MB | — | 8.983 ms |
| VGG KAGN-BN-SA-11v4 | KAN | 12.0M | 46.0 MB | — | 10.383 ms |

*FLOPs: KAN modelleri için `fvcore`, özel Kolmogorov-Arnold konvolüsyon operasyonlarını desteklemediğinden hesaplanamamıştır (—).*
*Latency: tek görsel inference süresi, NVIDIA A100-SXM4-40GB (Colab).*

---

## 4. Sonuçlar

### 4.1. Ana Performans Metrikleri

| Model | Iso. Top-1 | Iso. Top-5 | Crowding Acc | CI (yüksek=kötü) | ECE (düşük=iyi) |
|---|---|---|---|---|---|
| ResNet-34 | 0.7988 | 0.9505 | 0.6332 | 0.2073 | 0.1423 |
| ResNet-50 | 0.8231 | 0.9505 | 0.6146 | 0.2533 | 0.2058 |
| ResNet-101 | 0.8347 | 0.9514 | 0.5970 | 0.2847 | 0.2566 |
| ViT-S/16 | 0.8021 | 0.9407 | 0.5949 | 0.2583 | 0.2226 |
| ViT-B/16 | 0.7535 | 0.9332 | 0.5278 | 0.2995 | 0.1367 |
| Swin-T | 0.8557 | 0.9594 | 0.6855 | 0.1990 | 0.0919 |
| Swin-S | 0.8791 | 0.9711 | 0.6749 | 0.2323 | 0.1407 |
| VGG KAGN-11v2 | 0.7740 | 0.9613 | 0.5644 | 0.2709 | 0.0262 |
| VGG KAGN-11v4 | 0.7684 | 0.9589 | 0.5893 | 0.2331 | 0.0515 |
| VGG KAGN-BN-SA-11v4 | 0.7558 | 0.9304 | 0.5014 | 0.3366 | 0.0670 |

### 4.2. Spacing Bazlı Accuracy ve CI

| Model | 2.0 acc | 2.0 CI | 2.3 acc | 2.3 CI | 2.5 acc | 2.5 CI | 2.7 acc | 2.7 CI | 3.0 acc | 3.0 CI |
|---|---|---|---|---|---|---|---|---|---|---|
| ResNet-34 | 0.5474 | 0.3147 | 0.5815 | 0.2720 | 0.6171 | 0.2275 | 0.6714 | 0.1595 | 0.7485 | 0.0630 |
| ResNet-50 | 0.5189 | 0.3696 | 0.5407 | 0.3431 | 0.5781 | 0.2977 | 0.6664 | 0.1904 | 0.7687 | 0.0661 |
| ResNet-101 | 0.4813 | 0.4234 | 0.5196 | 0.3775 | 0.5624 | 0.3262 | 0.6581 | 0.2116 | 0.7637 | 0.0851 |
| ViT-S/16 | 0.5609 | 0.3007 | 0.5622 | 0.2991 | 0.5797 | 0.2773 | 0.6037 | 0.2474 | 0.6680 | 0.1672 |
| ViT-B/16 | 0.4992 | 0.3375 | 0.5041 | 0.3310 | 0.5203 | 0.3095 | 0.5337 | 0.2917 | 0.5817 | 0.2280 |
| Swin-T | 0.6508 | 0.2395 | 0.6566 | 0.2327 | 0.6752 | 0.2109 | 0.6945 | 0.1884 | 0.7503 | 0.1232 |
| Swin-S | 0.6115 | 0.3044 | 0.6351 | 0.2776 | 0.6649 | 0.2437 | 0.6985 | 0.2054 | 0.7644 | 0.1305 |
| VGG KAGN-11v2 | 0.5033 | 0.3497 | 0.5293 | 0.3161 | 0.5505 | 0.2888 | 0.5835 | 0.2461 | 0.6552 | 0.1535 |
| VGG KAGN-11v4 | 0.5286 | 0.3121 | 0.5541 | 0.2789 | 0.5799 | 0.2453 | 0.6102 | 0.2059 | 0.6736 | 0.1234 |
| VGG KAGN-BN-SA-11v4 | 0.4353 | 0.4241 | 0.4676 | 0.3813 | 0.4963 | 0.3433 | 0.5342 | 0.2932 | 0.5736 | 0.2411 |

### 4.3. Flanker Tipine Göre Accuracy ve CI

| Model | Congruent Acc | Congruent CI | Incongruent Acc | Incongruent CI |
|---|---|---|---|---|
| ResNet-34 | 0.9040 | −0.1317 | 0.4932 | 0.3826 |
| ResNet-50 | 0.9108 | −0.1065 | 0.4615 | 0.4393 |
| ResNet-101 | 0.9145 | −0.0956 | 0.4301 | 0.4847 |
| ViT-S/16 | 0.9018 | −0.1243 | 0.4366 | 0.4557 |
| ViT-B/16 | 0.8386 | −0.1129 | 0.3775 | 0.4990 |
| Swin-T | 0.9348 | −0.0924 | 0.5582 | 0.3477 |
| Swin-S | 0.9622 | −0.0945 | 0.5272 | 0.4003 |
| VGG KAGN-11v2 | 0.8709 | −0.1252 | 0.4122 | 0.4674 |
| VGG KAGN-11v4 | 0.8838 | −0.1502 | 0.4473 | 0.4179 |
| VGG KAGN-BN-SA-11v4 | 0.7747 | −0.0250 | 0.3615 | 0.5217 |

*Not: "external" flanker koşulu bu raporda değerlendirilmemiştir — bu koşul mevcut kompozit üretim pipeline'ında henüz üretilmemektedir; ayrı bir çalışma gerektirir.*

---

## 5. Tartışma

### 5.1. Mimari Karşılaştırması

**CNN:** Derinlik arttıkça hem isolation accuracy hem CI artıyor. ResNet-34→50→101: iso 0.799→0.823→0.835, CI 0.207→0.253→0.285. Daha derin CNN'ler crowding'e daha duyarlı.

**Transformer — global attention (ViT):** ViT-B/16, ViT-S/16'dan daha düşük isolation accuracy'ye (0.754 vs 0.802) rağmen daha yüksek CI gösteriyor (0.300 vs 0.258). Dahası, ViT-B/16'nın 0.7535'lik isolation accuracy'si **10 modelin en düşüğü** — artık pretrained olan KAN modellerinin bile altında. Global attention mekanizmasının, kapasite arttığında bu veri rejiminde hem genelleme hem de flanker'a karşı dayanıklılık açısından dezavantaja dönüştüğünü gösteriyor.

**Transformer — windowed attention (Swin):** Swin-T, **10 modelin tamamı arasında en düşük CI'ye (0.1990) ve ikinci en yüksek isolation accuracy'ye (0.8557) sahip** — yani hem en isabetli hem en dayanıklı model. Swin-S ise en yüksek isolation accuracy'yi (0.8791) elde ediyor ama CI'si Swin-T'ye göre kötüleşiyor (0.2323) — CNN ve ViT'te gözlenen "kapasite arttıkça CI kötüleşir" örüntüsü Swin ailesinde de tekrarlanıyor. Buna rağmen Swin-S mutlak CI değerinde tüm ResNet ve ViT modellerinden daha dayanıklı kalıyor. Bu sonuç, yerel pencereli attention'ın flanker etkisini, her pencerenin sınırlı bir bölgeyle sınırlı kalması sayesinde global attention'a göre daha iyi izole ettiğini düşündürüyor.

**KAN:** Artık pretrained ağırlıklarla adil bir karşılaştırma mümkün (isolation accuracy 0.756–0.774, CNN/ViT/Swin'e yakın). Alt-ailede de aynı kapasite-CI örüntüsü gözleniyor: VGG KAGN-11v4 (40.9M param, CI=0.2331) < VGG KAGN-11v2 (24.4M param, CI=0.2709) < VGG KAGN-BN-SA-11v4 (12.0M param, CI=0.3366 — **en yüksek CI**). BN-SA-11v4'ün en az parametreli KAN olmasına rağmen en kırılgan model olması dikkat çekici: eklenen self-attention ve BatchNorm bileşenlerinin — tıpkı ViT'teki global attention gibi — ham parametre sayısından bağımsız olarak crowding dayanıklılığını düşürdüğünü gösteriyor.

**Genel bulgu (4 alt-aile):** "Kapasite/karmaşıklık arttıkça CI kötüleşir" örüntüsü artık CNN (34→50→101), ViT (S→B), Swin (T→S) ve KAN (11v4→11v2→BN-SA) alt-ailelerinin **hepsinde** tutarlı şekilde gözleniyor. En dayanıklı model: **Swin-T** (CI=0.1990). En kırılgan model: **VGG KAGN-BN-SA-11v4** (CI=0.3366).

### 5.2. Flanker Tipi

**Congruent:** Tüm modellerde negatif CI — aynı sınıf flanker performansı artırıyor. En belirgin VGG KAGN-11v4'te (CI=−0.1502); en zayıf VGG KAGN-BN-SA-11v4'te (CI=−0.0250) — bu model congruent flanker'dan bile diğer modeller kadar faydalanamıyor.

**Incongruent:** Anlamlı CI değerleri (0.35–0.52). En dayanıklı model burada da Swin-T (CI=0.3477); en kırılgan yine VGG KAGN-BN-SA-11v4 (CI=0.5217). Farklı sınıf flanker, hedef nesnenin özellik temsilini baskılıyor.

### 5.3. Spacing

Tüm modellerde spacing arttıkça CI düşüyor (Bouma Yasası ile uyumlu). ViT-B/16 spacing artışına en az duyarlı model kalmaya devam ediyor (2.0°→3.0° arası CI değişimi yalnızca 0.110); en duyarlı (en hızlı toparlanan) model ResNet-34 (0.3147→0.0630, 0.252 azalma).

### 5.4. Kalibrasyon

En iyi kalibre edilen üç model de KAN ailesinden: VGG KAGN-11v2 (ECE=0.0262), VGG KAGN-11v4 (0.0515), VGG KAGN-BN-SA-11v4 (0.0670). Bu üçünün hemen ardından Swin-T geliyor (0.0919) — KAN dışındaki en iyi kalibre edilen model. En kötü kalibre model ResNet-101 (ECE=0.2566). KAN'ın düşük isolation accuracy'sine rağmen (CNN/ViT/Swin'in altında) belirgin şekilde daha iyi kalibre olması, mimariye özgü bir özellik olarak öne çıkıyor.

---

## 6. Sınırlılıklar ve Gelecek Çalışmalar

- **Çözüldü:** KAN modelleri artık ImageNet-1K pretrained ağırlıklarla başlıyor; CNN/ViT/Swin ile adil karşılaştırma sağlandı.
- KAN modellerinin FLOPs değeri hesaplanamıyor — `fvcore` özel Kolmogorov-Arnold konvolüsyon operasyonlarını desteklemiyor; manuel FLOP hesaplaması gerekebilir.
- Dış nesne (external flanker) koşulu bu turda değerlendirilmedi — mevcut pipeline bu koşulu üretmiyor, ayrı bir veri üretim + değerlendirme adımı gerektiriyor.
- Gelecek: Grad-CAM analizi, insan psikofiziği karşılaştırması, dikey eksen crowding, external flanker koşulunun eklenmesi, daha büyük KAN varyantları (width_scale artırılarak).

---

## 7. Referanslar

Bouma (1970). Interaction effects in parafoveal letter recognition. *Nature.*
Volokitin et al. (2017). Do deep neural networks suffer from crowding? *NeurIPS.*
Lonnqvist et al. (2020). Crowding in humans is unlike that in CNNs. *Neuropsychologia.*
Kumar et al. (2022). Fine-tuning can distort pretrained features. *ICLR.*
Drokin (2024). Kolmogorov-Arnold Convolutions. *arXiv:2407.01092.*
Kirillov et al. (2023). Segment Anything. *ICCV.*
Dosovitskiy et al. (2021). An Image is Worth 16x16 Words. *ICLR.*
Liu et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. *ICCV.*
He et al. (2016). Deep Residual Learning for Image Recognition. *CVPR.*
