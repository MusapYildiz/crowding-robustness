#!/bin/bash
# torch-conv-kan kurulum scripti
# PyPI'da mevcut değil, GitHub repo'su clone edilip manuel kurulur.

set -e

echo "=== torch-conv-kan kuruluyor ==="

if [ ! -d "torch-conv-kan" ]; then
    git clone https://github.com/IvanDrokin/torch-conv-kan.git
fi

cd torch-conv-kan && pip install -q -r requirements.txt && cd ..

# sys.path'e ekle (Python import için)
echo "$(pwd)/torch-conv-kan" >> $(python -c "import site; print(site.getsitepackages()[0])")/torch_conv_kan.pth

echo "=== Kurulum tamamlandı ==="
python -c "from kan_convs import KANConv2DLayer; print('KANConv2DLayer import: OK')"
