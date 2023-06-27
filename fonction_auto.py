import numpy as np
import pandas as pd
import sklearn
import pacmap
import matplotlib.pyplot as plt

import sklearn.cluster
from copy import deepcopy


def table_clusters(labels_X, labels_SHAP, seuil):
    if len(labels_X) != len(labels_SHAP) or max(labels_X) != max(labels_SHAP):
        print("BIP BIP NOOON")
        return
    taille = len(labels_X)
    n_clusters = max(labels_X) + 1
    table = np.zeros((n_clusters, n_clusters))
    table_2 = list(np.zeros((n_clusters, n_clusters)))
    table_2 = [list(table_2[i]) for i in range(n_clusters)]
    # print(table_2)
    l_X = deepcopy([[] for i in range(n_clusters)])
    l_SHAP = deepcopy([[] for i in range(n_clusters)])
    for i in range(taille):
        l_X[labels_X[i]].append(i)
        l_SHAP[labels_SHAP[i]].append(i)
    for i in range(n_clusters):
        for j in range(n_clusters):
            croisement = [x for x in l_X[i] if x in l_SHAP[j]]
            if len(croisement) < (len(labels_X) / n_clusters) * seuil:
                temp = -1
            else:
                temp = len([x for x in l_X[i] if x in l_SHAP[j]]) / max(
                    len(l_X[i]), len(l_SHAP[j])
                )
            table[i, j] = round(100 * temp, 2)
            table_2[i][j] = croisement
    table = pd.DataFrame(table)
    table.columns = ["ClustEV " + str(i) for i in range(n_clusters)]
    table.index = ["ClustEE " + str(i) for i in range(n_clusters)]
    table = table.astype(int)
    # cm = sns.light_palette("green", as_cmap=True)
    # table_couleurs = table.style.background_gradient(cmap=cm, axis=None)
    table_couleurs = table
    ligne = int(np.where(table.to_numpy() == table.max().max())[0])
    colonne = int(np.where(table.to_numpy() == table.max().max())[1])
    indices_clusters = [x for x in l_X[colonne] if i in l_SHAP[ligne]]
    # print(table_2[ligne][colonne])
    indices_clusters = table_2[ligne][colonne]
    # indices_clusters = l_SHAP[ligne]
    return table_couleurs, indices_clusters, table.max().max()


def new_indices(liste, liste_de_liste):
    l = []
    toutes_listes = [item for sublist in liste_de_liste for item in sublist]
    for i in range(len(liste)):
        compte = len([x for x in toutes_listes if x <= liste[i]])
        nombre = liste[i] + compte
        while nombre in toutes_listes:
            nombre += 1
        l.append(nombre)
    for i in l:
        if i in toutes_listes:
            print("AHHHH")
    return l


def create_liste(liste_de_liste, taille):
    ll = np.zeros(taille)
    for i in range(len(liste_de_liste)):
        for j in range(len(liste_de_liste[i])):
            ll[liste_de_liste[i][j]] = i
    return ll


def clustering_dyadique(val1, val2, n_clusters, seuil, show_all=False):
    l = []
    ll = np.zeros(len(val1))
    indices = range(len(val1))
    tot = 0
    taille = len(val1)
    for i in range(n_clusters - 1):
        model1 = sklearn.cluster.KMeans(
            n_clusters=n_clusters - i, random_state=9, n_init="auto"
        )
        model1.fit(val1)
        model2 = sklearn.cluster.KMeans(
            n_clusters=n_clusters - i, random_state=9, n_init="auto"
        )
        model2.fit(val2)
        calcul = table_clusters(model1.labels_, model2.labels_, seuil)
        indices_clusters = calcul[1]
        indices_pour_couleur = deepcopy(indices_clusters)
        indices_a_rajouter = []
        for ii in range(len(indices_clusters)):
            indices_a_rajouter.append(indices[indices_clusters[ii]])
        l.append(indices_a_rajouter)
        indices = [i for i in indices if i not in indices_a_rajouter]
        tot += len(indices_clusters)
        pourcent_clu = round(100 * len(indices_clusters) / taille, 2)
        pourcent_tot = round(100 * tot / taille, 2)
        # print(
        #    f"Cluster #{i+1} trouvé : ({len(indices_clusters)} points, {pourcent_clu}% du dataset, {pourcent_tot}% en cumulé)"
        # )
        color = []
        for i in range(len(val1)):
            if i in indices_pour_couleur:
                color.append(1)
            else:
                color.append(0)
        if show_all:
            pass
        val1 = val1[[i for i in range(len(val1)) if i not in indices_pour_couleur]]
        val2 = val2[[i for i in range(len(val2)) if i not in indices_pour_couleur]]
    tous_indices = [item for sublist in l for item in sublist]
    indices_manquants = [i for i in range(taille) if i not in tous_indices]
    l.append(indices_manquants)
    tot += len(indices_manquants)
    pourcent_clu = round(100 * len(indices_manquants) / taille, 2)
    pourcent_tot = round(100 * tot / taille, 2)
    # print(
    #    f"Cluster #{n_clusters} trouvé : ({len(indices_manquants)} points, {pourcent_clu}% du dataset, {pourcent_tot}% en cumulé)"
    # )
    return l, create_liste(l, taille)
