# 🚀 Guide d'optimisation des performances d'Antakia

Ce guide vous aide à résoudre les problèmes de lenteur d'Antakia.

## 🚨 Problèmes identifiés

La lenteur d'Antakia est principalement due à :
1. **Calculs SHAP très lents** au démarrage
2. **Double projection PaCMAP** coûteuse en temps
3. **Configuration par défaut non-optimisée**

## ⚡ Solutions rapides

### 1. Utilisation avec configuration rapide

```python
# Avant d'importer Antakia, appliquer la config rapide
from antakia.performance_config import apply_fast_config
apply_fast_config()

from antakia.antakia import AntakIA
# Le reste de votre code...
```

### 2. Configurations disponibles

```python
from antakia import performance_config as perf

# Pour un démarrage ultra-rapide (recommandé)
perf.apply_fast_config()      # PCA + LIME + 2000 points max

# Pour un équilibre performance/qualité  
perf.apply_medium_config()    # UMAP + LIME + 3000 points max

# Pour privilégier la qualité (lent)
perf.apply_quality_config()   # PaCMAP + SHAP + 5000 points max

# Pour les très gros datasets
perf.apply_minimal_config()   # PCA + LIME + 1000 points max
```

### 3. Configuration manuelle via variables d'environnement

```bash
export DEFAULT_PROJECTION=PCA           # Au lieu de PaCMAP
export DEFAULT_EXPLANATION_METHOD=2     # LIME (2) au lieu de SHAP (1)  
export MAX_DOTS=2000                   # Limiter les points affichés
```

## 📊 Temps de démarrage estimés

| Configuration | Temps (dataset 1000 lignes) | Temps (dataset 10k lignes) |
|---------------|------------------------------|----------------------------|
| **Rapide**    | 5-10 secondes               | 15-30 secondes             |
| **Équilibrée**| 15-30 secondes              | 1-3 minutes                |
| **Qualité**   | 1-5 minutes                 | 10-30 minutes              |
| **Ancienne**  | 5-30 minutes                | 1-3 heures                 |

## 🔧 Pour les développeurs

### Modifications apportées :

1. **config.py** : PCA par défaut au lieu de PaCMAP
2. **explanation_values.py** : Mode lazy pour éviter les calculs SHAP au démarrage  
3. **gui.py** : Gestion du fallback quand pas d'explications
4. **performance_config.py** : Configurations prêtes à l'emploi

### Si vous voulez restaurer l'ancien comportement :

```python
from antakia.performance_config import apply_quality_config
apply_quality_config()  # Comportement original mais lent
```

## 🆘 En cas de problème

Si vous rencontrez encore des problèmes :

1. Vérifiez votre version Python (>=3.10, <3.12)
2. Redémarrez votre kernel Jupyter
3. Utilisez la configuration minimale pour très gros datasets
4. Envisagez de sous-échantillonner vos données pour les premiers tests

## 📈 Monitorer les performances 

Les temps de calcul sont loggés automatiquement. Vérifiez les logs pour identifier les goulots d'étranglement restants.
