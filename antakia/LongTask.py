"""
Class to compute long tasks in a separate thread.
"""

import time
import pandas as pd
import shap
import ipyvuetify as v
import lime
import lime.lime_tabular
import numpy as np
import threading

from abc import ABC, abstractmethod

class LongTask(ABC):

    def __init__(self, X, X_all, model):
        self.X = X
        self.X_all = X_all
        self.model = model

        self.progress = 0
        self.progress_widget = v.Textarea(v_model=0)
        self.text_widget = v.Textarea(v_model=None)
        self.done_widget = v.Textarea(v_model=True)
        self.value = None
        self.thread = None
    
    @abstractmethod
    def compute(self):
        pass
    @abstractmethod
    def start_thread(self):
        pass

    def generation_texte(self, i, tot, time_init, progress):
        progress = float(progress)
        # allows to generate the progress text of the progress bar
        time_now = round((time.time() - time_init) / progress * 100, 1)
        minute = int(time_now / 60)
        seconde = time_now - minute * 60
        minute_passee = int((time.time() - time_init) / 60)
        seconde_passee = int((time.time() - time_init) - minute_passee * 60)
        return (
            str(round(progress, 1))
            + "%"
            + " ["
            + str(i + 1)
            + "/"
            + str(tot)
            + "] - "
            + str(minute_passee)
            + "m"
            + str(seconde_passee)
            + "s (temps estimé : "
            + str(minute)
            + "min "
            + str(round(seconde))
            + "s)"
        )

class compute_SHAP(LongTask):
    def compute(self):
        self.progress = 0
        self.done_widget.v_model = "primary"
        self.text_widget.v_model = None
        time_init = time.time()
        explainer = shap.Explainer(self.model.predict, self.X_all)
        shap_values = pd.DataFrame().reindex_like(self.X)
        j = list(self.X.columns)
        for i in range(len(j)):
            j[i] = j[i] + "_shap"
        for i in range(len(self.X)):
            shap_value = explainer(self.X[i : i + 1], max_evals=1400)
            shap_values.iloc[i] = shap_value.values
            self.progress += 100 / len(self.X)
            self.progress_widget.v_model = self.progress
            self.text_widget.v_model = self.generation_texte(i, len(self.X), time_init, self.progress_widget.v_model)
        shap_values.columns = j
        self.value = shap_values
        self.done_widget.v_model = "success"
        return shap_values
    
    def start_thread(self):
        self.thread = threading.Thread(target=self.compute)
        self.thread.start()

class compute_LIME(LongTask):
    
    def compute(self):
        self.done_widget.v_model = "primary"
        self.progress_widget.v_model = 0
        self.text_widget.v_model = None
        time_init = time.time()
        explainer = lime.lime_tabular.LimeTabularExplainer(np.array(self.X_all), feature_names=self.X.columns, class_names=['price'], verbose=False, mode='regression')
        N = len(self.X)
        LIME = pd.DataFrame(np.zeros((N, self.X.shape[-1])))
        l = []
        for j in range(N):
            l = []
            exp = explainer.explain_instance(
                self.X.values[j], self.model.predict
            )
            l = []
            taille = self.X.shape[-1]
            for ii in range(taille):
                exp_map = exp.as_map()[0]
                l.extend(exp_map[ii][1] for jj in range(taille) if ii == exp_map[jj][0])
            LIME.iloc[j] = l
            self.progress_widget.v_model  += 100 / len(self.X)
            self.text_widget.v_model = self.generation_texte(j, len(self.X), time_init, self.progress_widget.v_model)
        j = list(self.X.columns)
        for i in range(len(j)):
            j[i] = j[i] + "_shap"
        LIME.columns = j
        self.value = LIME
        self.done_widget.v_model = "success"
        return LIME
    
    def start_thread(self):
        self.thread = threading.Thread(target=self.compute)
        self.thread.start()