"""
Configuration optimisée pour les performances d'Antakia.

Ce fichier contient des configurations prêtes à l'emploi pour optimiser 
les performances selon différents scénarios d'usage.
"""

import os
import multiprocessing

def apply_fast_config():
    """
    Configuration pour un démarrage rapide.
    Utilise PCA au lieu de PaCMAP et évite les calculs SHAP au démarrage.
    """
    os.environ['DEFAULT_PROJECTION'] = 'PCA'
    os.environ['DEFAULT_EXPLANATION_METHOD'] = '2'  # LIME (plus rapide que SHAP)
    os.environ['MAX_DOTS'] = '2000'  # Réduire le nombre de points affichés
    print("✅ Configuration rapide activée : PCA + LIME + limite de points")

def apply_medium_config():
    """
    Configuration équilibrée entre performance et qualité.
    """
    os.environ['DEFAULT_PROJECTION'] = 'UMAP'  
    os.environ['DEFAULT_EXPLANATION_METHOD'] = '2'  # LIME
    os.environ['MAX_DOTS'] = '3000'
    print("✅ Configuration équilibrée activée : UMAP + LIME")

def apply_quality_config():
    """
    Configuration pour privilégier la qualité au détriment de la rapidité.
    """
    os.environ['DEFAULT_PROJECTION'] = 'PaCMAP'
    os.environ['DEFAULT_EXPLANATION_METHOD'] = '1'  # SHAP
    os.environ['MAX_DOTS'] = '5000'
    print("⚠️  Configuration qualité activée : PaCMAP + SHAP (démarrage lent)")

def apply_minimal_config():
    """
    Configuration minimaliste pour les très gros datasets.
    """
    os.environ['DEFAULT_PROJECTION'] = 'PCA'
    os.environ['DEFAULT_EXPLANATION_METHOD'] = '2'  # LIME
    os.environ['MAX_DOTS'] = '1000'  # Très peu de points
    print("⚡ Configuration minimaliste activée : performances maximales")

def apply_m2_max_config():
    """
    Configuration optimisée spécifiquement pour MacBook Pro M2 Max.
    Exploite les 12 cœurs CPU et 32GB RAM.
    """
    # Configuration de base rapide
    os.environ['DEFAULT_PROJECTION'] = 'PCA'
    os.environ['DEFAULT_EXPLANATION_METHOD'] = '2'  # LIME
    os.environ['MAX_DOTS'] = '3000'  # Plus de points grâce à la RAM
    
    # Optimisations spécifiques M2 Max
    os.environ['OMP_NUM_THREADS'] = '12'  # Utiliser tous les cœurs
    os.environ['MKL_NUM_THREADS'] = '12'   # Optimisation Intel MKL
    os.environ['OPENBLAS_NUM_THREADS'] = '12'  # Optimisation OpenBLAS
    os.environ['VECLIB_MAXIMUM_THREADS'] = '12'  # Optimisation Apple Accelerate
    
    # Configuration mémoire pour 32GB RAM
    os.environ['ATK_MEMORY_LIMIT'] = '24'  # GB (laisser 8GB pour le système)
    os.environ['ATK_CHUNK_SIZE'] = '10000'  # Taille des chunks pour le traitement
    
    print("🚀 Configuration M2 Max activée :")
    print("   • 12 cœurs CPU utilisés")
    print("   • 24GB RAM alloués")
    print("   • PCA + LIME + 3000 points max")
    print("   • Optimisations BLAS/Accelerate activées")

def apply_m2_max_parallel_config():
    """
    Configuration M2 Max avec parallélisme maximal pour gros datasets.
    """
    # Configuration de base
    apply_m2_max_config()
    
    # Optimisations parallèles avancées
    os.environ['ATK_PARALLEL_PROCESSING'] = 'True'
    os.environ['ATK_NUM_WORKERS'] = '10'  # 10 workers (laisser 2 cœurs pour le système)
    os.environ['ATK_BATCH_SIZE'] = '5000'  # Taille de batch optimisée
    os.environ['ATK_USE_NUMPY_PARALLEL'] = 'True'
    
    print("⚡ Mode parallèle M2 Max activé :")
    print("   • Traitement parallèle activé")
    print("   • 10 workers parallèles")
    print("   • Batching optimisé")

# Configuration par défaut - mode rapide
apply_fast_config()
