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
  echo "Fallback git: sope-rules-antakia@v0.1.1"
  pip install "sope-rules-antakia @ git+https://github.com/AI-vidence/sope-rules-antakia.git@v0.1.1"
}

install_antakia_core() {
  if pip install "antakia-core>=0.4.9,<0.5" 2>/dev/null; then
    echo "antakia-core installé depuis PyPI"
    return
  fi
  echo "Fallback git: antakia-core release/0.4.9"
  git clone --depth 1 --branch release/0.4.9 \
    https://github.com/AI-vidence/antakia-core.git /tmp/antakia-core
  pip install /tmp/antakia-core --no-deps
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
