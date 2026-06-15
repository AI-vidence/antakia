#!/usr/bin/env bash
# Contournement résolution pip/poetry : antakia-core exige skope-rules-temp (numpy<2)
# alors que shap>=0.50 exige numpy>=2. On installe les binaires numba puis les paquets
# PyPI critiques sans dépendances transitives, puis antakia en editable (--no-deps).
set -euo pipefail

pip install -U pip

pip install "numba>=0.59" "llvmlite>=0.42" --only-binary=numba,llvmlite
pip install \
  "numpy>=2" \
  "shap==0.51.0" \
  "pandas>=2.2,<3" \
  "scikit-learn>=1.4,<2" \
  scipy \
  "interpret>=0.6" \
  "lime>=0.2,<0.3" \
  "pygam>=0.9" \
  "umap-learn>=0.5.5,<0.6"

pip install "antakia-core==0.4.8" "skope-rules-temp==0.2.6" "antakia-ac==0.2.14" --no-deps

pip install \
  "ipyvuetify<1.9" \
  ipywidgets \
  "plotly<=5.19" \
  python-dotenv \
  requests \
  pdpbox \
  kaleido \
  xhtml2pdf

pip install -e . --no-deps
