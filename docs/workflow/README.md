# Workflow AntakIA — Notebooks par étape

Ce dossier contient **un notebook par étape** du workflow AntakIA, numérotés chronologiquement. Chaque étape peut être travaillée indépendamment une fois les étapes précédentes comprises.

## Vue d'ensemble

```
Étape 0   Données & Modèle prédictif           (→ dossier examples/)        [David]
Étape 1   Valeurs de Shapley & indices          01_shap_values.ipynb         [PSC]
Étape 2   Interactions & indices coopératifs    02_interactions.ipynb        [PSC ★]
Étape 3   Parcellisation (8 méthodes)           03_parcellation.ipynb        [PSC]
Étape 4   Tessellation (distillation régionale) 04_tessellation.ipynb        [PSC]
Étape 5   Personas & Contrefactuels             05_counterfactuals.ipynb     [PSC]
Étape 6   Narration globale (LLM)               (→ examples/06_*)           [David]
Étape 7   Monitoring de dérive                  (→ examples/07_*)           [David]
Étape 8   Observabilité                         (→ examples/08_*)           [David]
Étape 9   Détection/Correction de biais         (→ repo ARTIST)             [David]
Étape 10  Pipeline complet                      (→ examples/10_*)           [David]
```

### Logique du workflow

```
Étape 0 : Données + Modèle                       ← en amont
    ↓
Étape 1 : Pourquoi ce score ? (Shapley & co.)     ← attribution locale par 5 indices
    ↓
Étape 2 : Quelles synergies ? (Interactions)      ← effets conjoints, groupes, hiérarchie
    ↓
Étape 3 : Quels groupes ? (Parcellisation)        ← 8 méthodes dont GRANITE & Ensemble
    ↓
Étape 4 : Quelles règles ? (Tessellation)         ← modèle local par région
    ↓
Étape 5 : Qui est-ce ? (Persona + contrefactuel)  ← explication humaine par tesselle
    ↓
Étape 6 : Que fait le modèle ? (Narration LLM)    ← carte complète de toutes les tesselles
    ↓
    ↓  ─── Perspective : approches causales (autre PSC) ───
    ↓
Étape 7-10 : Monitoring, Observabilité, Biais, Pipeline complet
```

---

## Inventaire des méthodes implémentées

Avant de détailler chaque étape, voici ce qui est **déjà disponible** dans le code :

### Indices explicatifs (`fast_tree_explainer`)

| Indice | Ordre | Description | Complexité |
|--------|-------|-------------|------------|
| **Shapley** | 1, 2 | Attribution additive (axiomes d'efficience, symétrie, linéarité, dummy) | O(T·L·D²) par sample |
| **Banzhaf** | 1 | Pondération uniforme (relaxe l'efficience) — plus robuste si features corrélées | O(T·L·D²) |
| **Sobol** | 1 | Sensibilité globale (proportion de variance) — vision globale, pas locale | O(N·T·L·D²) |
| **Owen** | 1 | Shapley respectant une structure de groupes — attribution intra/inter groupe | O(G!·T·L·D²) |
| **Winter** | 1 | Owen hiérarchique — groupes emboîtés | O(G!·T·L·D²) |

**Backends** : MLX (Apple Silicon GPU), C++ multi-thread, CUDA (stub)

### Méthodes de parcellisation (`antakia/parcellation/`)

| Méthode | Classe | Principe |
|---------|--------|----------|
| **GRANITE** | `GraniteParcellation` | Guidée par les interactions Shapley (arXiv:2601.22771) |
| **HDBSCAN Dyadic** | `HDBSCANDyadicParcellation` | Clustering densité sur espace dyadique VS×ES |
| **ToMATo Dyadic** | `ToMaToDyadicParcellation` | Analyse topologique (modes de persistance) |
| **MGE** | `MGEParcellation` | Multi-Granularity Ensemble |
| **Consensus** | `ConsensusParcellation` | Consensus multi-vues |
| **Ensemble** | `EnsembleParcellation` | Meta-méthode : noyaux stables à partir de HDBSCAN+ToMATo+MGE+Consensus |
| **Semi-supervisé** | `IVISSemiSupervisedParcellation` | Ensemble + projection UMAP/IVIS semi-supervisée |
| **AntakIA Native** | `AntakiaNativeParcellation` | Auto-clustering ToMATo sur projection 3D |

### Tessellation (`antakia/tessellation/`)

- **TessellationEngine** : raffinement itératif des tesselles
- **Pureté cartésienne** : score de qualité des régions
- **Modèles locaux** : Ridge, GAM, arbre de décision par tesselle
- **Règles Skope** : extraction de règles IF-THEN
- **Description LLM** : génération de descriptions narratives

### Contrefactuels (`antakia/tessellation/counterfactuals.py`)

- **DiCE** : Diverse Counterfactual Explanations (Microsoft)
- **Plus proche voisin** : fallback quand DiCE indisponible
- **Gradient-based** : méthode alternative
- **Archétype** : point le plus proche du barycentre

---

## Détail des étapes 1-6

### Étape 1 — Valeurs de Shapley & indices (`01_shap_values.ipynb`)

**Objectif** : Calculer et comparer 5 indices d'attribution pour un même modèle.

**API clé** :
```python
from fast_tree_explainer import FastTreeExplainer

explainer = FastTreeExplainer(model)

sv    = explainer.explanation_values(X, index="shapley")   # (N, M) additif exact
bv    = explainer.explanation_values(X, index="banzhaf")   # (N, M) pondération uniforme
sobol = explainer.explanation_values(X, index="sobol")     # (M,)  sensibilité globale
```

**Ce qui est implémenté** : Les 5 indices sur 3 backends (MLX, C++, fallback SHAP).

**Pistes de travail PSC** :
1. **Comparer Shapley vs Banzhaf** sur features corrélées : la propriété d'efficience (Shapley somme à la prédiction) est-elle un avantage ou un biais quand les features sont redondantes ?
2. **Sobol vs Shapley** : Sobol est global (un vecteur pour tout X), Shapley est local (un vecteur par sample). Sur quels jeux de données divergent-ils le plus ?
3. **Benchmark de performance** : mesurer le speedup MLX vs C++ vs SHAP standard sur California Housing (20,640 points × 8 features) et sur des jeux plus larges.

---

### Étape 2 — Interactions & indices coopératifs (`02_interactions.ipynb`)

**Objectif** : Comprendre les effets conjoints entre variables via les interactions d'ordre 2, et explorer les indices de groupe (Owen, Winter).

**API clé** :
```python
# Interactions d'ordre 2 — matrice N×M×M
interactions = explainer.explanation_values(X, index="shapley", order=2)

# Owen values — Shapley respectant des groupes sémantiques
groups = [[0,1,2], [3,4,5], [6,7,8,9]]   # max 6-8 groupes (G! permutations)
owen = explainer.explanation_values(X, index="owen", groups=groups)

# Winter values — Owen hiérarchique
hierarchy = [[0,1], [2]]   # super-groupes
winter = explainer.explanation_values(X, index="winter",
                                       groups=groups, hierarchy=hierarchy)
```

**Ce qui est implémenté** : Interactions ordre 2, Owen, Winter (C++ et MLX).

**Pistes de travail PSC (CŒUR DU PROJET)** :

1. **Extension ordre >2** : L'implémentation actuelle s'arrête à l'ordre 2 (matrice N×M×M). Proposer un algorithme pour l'ordre 3 (tenseur N×M×M×M). Contraintes : complexité O(M³) en mémoire — comment approximer ? Échantillonnage aléatoire ? Décomposition tensorielle ?

2. **Groupes sémantiques pour Owen** : Au lieu de groupes aléatoires, construire des groupes à partir de :
   - la matrice de corrélation des features
   - la matrice d'interactions Shapley elle-même (clustering des features qui interagissent)
   - la connaissance métier (variables financières, démographiques, géographiques...)
   - Comparer la qualité des attributions Owen avec chaque stratégie de groupement.

3. **Winter et hiérarchies naturelles** : Si les features ont une structure arborescente naturelle (ex: catégories → sous-catégories), Winter capture mieux l'attribution. Construire cette hiérarchie automatiquement à partir du dendrogramme des corrélations.

4. **Synergie / Redondance / Indépendance** : À partir de la matrice d'interactions, décomposer chaque paire (i,j) en effet synergique (conjoint > séparé), redondant (conjoint < séparé), ou indépendant. Quel lien avec la corrélation ?

**Attention performance** :
- Interactions ordre 2 : sous-échantillonner à 200-500 points si M > 50
- Owen : G! permutations → garder G ≤ 6 (6! = 720, OK ; 8! = 40320, lent)
- Winter : même contrainte que Owen sur le nombre de groupes

---

### Étape 3 — Parcellisation (`03_parcellation.ipynb`)

**Objectif** : Segmenter les données dans l'espace dyadique VS×ES en utilisant les 8 méthodes disponibles.

**API clé** :
```python
from antakia.parcellation.factory import create_parcellation

# Méthodes disponibles
methods = [
    "hdbscan-dyadic",   # Clustering densité
    "tomato-dyadic",    # Analyse topologique
    "mge",              # Multi-Granularity Ensemble
    "consensus",        # Consensus multi-vues
    "granite",          # Guidée par les interactions ★
    "ensemble",         # Noyaux stables (stabilité)
    "semisup",          # Semi-supervisé (IVIS/UMAP)
    "native",           # AntakIA natif (ToMATo 3D)
]

# Exemple : GRANITE (utilise les interactions Shapley)
parc = create_parcellation("granite", X=X, shap_values=sv, interactions=interactions)
regions = parc.fit()

# Ensemble : extrait les noyaux stables
ensemble = create_parcellation("ensemble", X=X, shap_values=sv)
regions = ensemble.fit()
stability_matrix = ensemble.stability_matrix_     # N×N co-association
stability_scores = ensemble.stability_scores_     # (N,) score 0-1 par point
```

**Ce qui est implémenté** : Les 8 méthodes ci-dessus, avec stabilité et semi-supervisé.

**Pistes de travail PSC** :

1. **GRANITE et interactions** : GRANITE utilise les interactions Shapley pour guider la parcellisation. Comparer : les régions sont-elles plus "pures" (homogènes en explication) quand on utilise les interactions d'ordre 2 vs seulement l'ordre 1 ? Et si on avait l'ordre 3 ?

2. **Ensemble et stabilité** : L'`EnsembleParcellation` combine HDBSCAN + ToMATo + MGE + Consensus et extrait des "noyaux stables" (points toujours classés ensemble). Étudier :
   - La matrice de stabilité N×N : visualiser comme un graphe (nœuds = points, arêtes = co-clustering fréquent)
   - Les scores de stabilité par point : les points instables sont-ils des cas ambigus pour le modèle ? Des cas "frontière" ?
   - Le seuil de stabilité : comment choisir `stability_threshold` ? Impact sur le nombre et la taille des régions ?

3. **Semi-supervisé** : `IVISSemiSupervisedParcellation` projette les données avec UMAP/IVIS en utilisant la supervision des noyaux stables. Explorer :
   - L'impact du `supervision_weight` : plus on supervise, plus les régions se "cristallisent"
   - Peut-on utiliser les interactions d'ordre >2 comme signal de supervision additionnel ?

4. **Comparaison systématique** : Sur California Housing, comparer les 8 méthodes sur :
   - Nombre de régions
   - Pureté cartésienne (homogénéité score × explication)
   - Score de Rand ajusté entre méthodes (convergent-elles ?)
   - Temps de calcul

---

### Étape 4 — Tessellation (`04_tessellation.ipynb`)

**Objectif** : Remplacer le modèle black-box par une collection de modèles locaux simples (distillation régionale).

**Ce qui est implémenté** :
- `TessellationEngine` avec raffinement itératif
- Score de pureté cartésienne (homogénéité score × explication)
- Modèles locaux : Ridge, GAM, arbre de décision
- Règles Skope-Rules (IF-THEN)

**Pistes de travail PSC** :

1. **Interaction-guided purity** : La pureté actuelle mesure l'homogénéité des Shapley dans chaque tesselle. Si on remplace les Shapley par les interactions (tenseur M×M), la subdivision itérative produit-elle des tesselles plus interprétables ?

2. **Choix du modèle local** : Comparer Ridge vs arbre vs GAM par tesselle. Quelle mesure de fidélité (R², MSE local vs global) ? Le meilleur modèle local est-il toujours le même pour toutes les tesselles ?

3. **Skope-Rules et lisibilité** : Les règles Skope sont parfois longues et opaques. Explorer :
   - La simplification par élagage
   - La complémentarité avec les Shapley dominants : la règle Skope capture-t-elle les mêmes variables que les top-3 Shapley de la tesselle ?

---

### Étape 5 — Personas & Contrefactuels (`05_counterfactuals.ipynb`)

**Objectif** : Rendre chaque tesselle compréhensible pour un humain.

**Ce qui est implémenté** :
- Archétype = point le plus proche du barycentre de la tesselle
- DiCE = changement minimal pour basculer dans une autre tesselle
- `LLMExplainer` pour génération de descriptions narratives

**Méthode en 3 temps** :
1. **Archétype** (barycentre) → "voici le profil type de cette région"
2. **Contrefactuel** (DiCE) → "pour passer du groupe A au groupe B, modifier X de a à b"
3. **Persona LLM** → "Cette tesselle représente des artisans plombiers urbains, sensibles au délai de livraison..."

**Pistes de travail PSC** :

1. **Qualité des contrefactuels** : DiCE produit des contrefactuels "diversifiés". Mais sont-ils réalistes ? Comparer les contrefactuels DiCE aux plus proches voisins de l'archétype dans une tesselle voisine. Le changement est-il actionnable ?

2. **Persona enrichie par les interactions** : L'archétype capture le "centre" de la tesselle. Les interactions capturent les synergies. Un persona plus riche dirait non seulement "cette personne gagne X€ et habite à Y" mais aussi "le lien entre revenu et localisation est le principal moteur de son score".

3. **Prompt engineering LLM** : Le `LLMExplainer` existe. Expérimenter différents prompts pour produire des descriptions de personas lisibles par un non-technicien. Évaluer la qualité avec des métriques simples (longueur, exhaustivité des features top-K, cohérence avec les règles Skope).

---

### Étape 6 — Narration globale (`examples/06_llm_explanation_demo.ipynb`)

**Objectif** : Produire une description narrative de TOUTES les tesselles et parcelles pour qu'un humain comprenne globalement le fonctionnement du modèle.

**Distinction avec l'étape 5** : L'étape 5 = loupe sur une tesselle. L'étape 6 = carte complète de toutes les tesselles.

**Entrées** : toutes les tesselles + personas + règles + interactions
**Sorties** : texte structuré type "Le modèle identifie 4 grands profils : ..."

---

## Perspective : approches causales

> Les approches causales (DAG, do-calculus, interventions) font l'objet d'un **PSC séparé**. Dans le code actuel, les interactions Shapley sont interprétées comme un **signal de causalité** (des variables qui interagissent fortement ont probablement un lien causal), mais aucune inférence causale formelle n'est implémentée. L'articulation entre les deux PSC se fera naturellement :
>
> - **PSC Interactions** (ce document) : fournit les interactions d'ordre >2 et les groupes structurels
> - **PSC Causal** : utilise ces structures comme squelette d'un DAG causal, et teste la causalité via des interventions (do-calculus) et des tests d'indépendance conditionnelle

---

## Pour les étudiants PSC

**Parcours recommandé** :
1. Commencer par `01_shap_values.ipynb` — comprendre les 5 indices
2. Passer à `02_interactions.ipynb` — c'est le cœur du sujet PSC
3. Consulter `03_parcellation.ipynb` — voir comment les interactions alimentent GRANITE et l'Ensemble
4. `04_tessellation.ipynb` — comprendre la distillation régionale et la pureté
5. `05_counterfactuals.ipynb` — comprendre les personas (archétype + contrefactuel + LLM)

Les étapes 3-5 montrent comment les améliorations des interactions (étape 2) se propagent en aval vers des explications plus nettes.

**Documentation complémentaire** :
- `antakia-core/docs/explainability/cooperative_indices.md` — Guide scientifique des indices
- `fast_tree_explainer/docs/API_REFERENCE.md` — Référence API complète
- `fast_tree_explainer/docs/BACKENDS.md` — Détails des backends GPU/C++

---

**Date** : 2026-01-23
