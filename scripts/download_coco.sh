#!/bin/bash
# COCO 2017 train images + annotations indir
# Colab'da çalıştır: bash scripts/download_coco.sh

set -e

COCO_DIR="./coco"
mkdir -p "$COCO_DIR/images"
mkdir -p "$COCO_DIR/annotations"

echo "=== COCO 2017 train images indiriliyor (~18GB) ==="
wget -c http://images.cocodataset.org/zips/train2017.zip -O "$COCO_DIR/train2017.zip"
unzip -q "$COCO_DIR/train2017.zip" -d "$COCO_DIR/images/"
rm "$COCO_DIR/train2017.zip"

echo "=== COCO 2017 annotations indiriliyor ==="
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip \
     -O "$COCO_DIR/annotations.zip"
unzip -q "$COCO_DIR/annotations.zip" -d "$COCO_DIR/"
rm "$COCO_DIR/annotations.zip"

echo "=== Tamamlandı ==="
echo "Görsel dizini : $COCO_DIR/images/train2017"
echo "Annotation    : $COCO_DIR/annotations/instances_train2017.json"
