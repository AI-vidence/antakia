<img src="assets/logo_ai-vidence.png" width="200">
<br>
<img src="assets/logo_antakia.png" width="200">

# AntakIA

Ce projet propose la méthode d'analyse XAI AntakIA de façon intégrée à un notebook Jupyter.
L'UI repose sur des ipywidgets.

# Installation avec `pip`

- Créer un environnement virtuel à partir de Python3.10 : `python3.10 -m venv .`
- Dans cet environement faire `pip install -r requirements.txt`
- Installer `skope-rules` depuis son repo GitHub : `pip install git+https://github.com/scikit-learn-contrib/skope-rules.git`

# Installation avec Docker

- Assurer vous d'avoir un Docker engine qui tourne sur votre machine
- `docker build -t antakia .`
- `docker run -p 8888:8888 antakia`

# Fonctionnement

- Testez les notebooks fournis (extenson `.ipynb`)

# Notes sur les librairies utilisées

## SHAP

- Ne fonctionne pas avec numpy >= 1.24 https://github.com/slundberg/shap/issues/2911
- D'où la version 1.23 dans `requirements.txt`

## Numba

- Directives de compilation Numba dépréciées https://github.com/slundberg/shap/issues/2909
- Il faut attendre que ce merge soit accepté https://github.com/dsgibbons/shap/pull/9
- En attendant, les exceptions sont "catchées" dans le code

## Skope rules

- Le repo https://github.com/scikit-learn-contrib/skope-rules est OK mais la version de PyPi non
- Il faut l'nstaller depuis GitHub : `pip install git+https://github.com/scikit-learn-contrib/skope-rules.git`