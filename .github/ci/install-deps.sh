#!/usr/bin/env bash
# Installe AntakIA et ses dépendances pour la CI.
set -euo pipefail

pip install -U pip
pip install "numba>=0.59" "llvmlite>=0.42" --only-binary=numba,llvmlite

install_sope_rules() {
  if pip install "sope-rules-antakia>=0.1.1" 2>/dev/null; then
    echo "sope-rules-antakia installé depuis PyPI"
    return
  fi
  if [ -d _deps/sope-rules-antakia ]; then
    echo "Fallback checkout local: sope-rules-antakia"
    pip install ./_deps/sope-rules-antakia
    return
  fi
  echo "ERREUR: sope-rules-antakia absent de PyPI et pas de checkout _deps/"
  exit 1
}

install_antakia_core() {
  if pip install "antakia-core>=0.4.9,<0.5" 2>/dev/null; then
    echo "antakia-core installé depuis PyPI"
    return
  fi
  if [ -d _deps/antakia-core ]; then
    echo "Fallback checkout local: antakia-core"
    pip install -e _deps/antakia-core --no-deps
    return
  fi
  echo "ERREUR: antakia-core>=0.4.9 absent de PyPI et pas de checkout _deps/"
  exit 1
}

install_sope_rules

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

install_antakia_core

pip install "antakia-ac==0.2.14" --no-deps

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
