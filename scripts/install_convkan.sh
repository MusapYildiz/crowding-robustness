#!/bin/bash
# torch-conv-kan GitHub'dan kurulum
# PyPI'da mevcut değil, direkt repo'dan kurulur.

set -e

echo "=== torch-conv-kan kuruluyor ==="
pip install -q git+https://github.com/IvanDrokin/torch-conv-kan.git
echo "=== Kurulum tamamlandı ==="

# Test
python -c "from kan_convs import KANConv2DLayer; print('KANConv2DLayer import OK')"
