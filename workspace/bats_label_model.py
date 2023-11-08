import pickle
from snorkel.labeling.model import LabelModel
import numpy as np
from pruning_lfs import prune_lfs
from snorkel.labeling import PandasLFApplier
import pandas as pd

class BatsModel:
    def __init__(self, ds_name):
        simp_path = f"/workspace/datasets/ds_labels/{ds_name}_simp_labels.pkl"        
        src_path = f"/workspace/datasets/ds_labels/{ds_name}_src_labels.pkl"  

        self.simp_labels = pickle.load(open(simp_path, "rb"))
        self.src_labels = pickle.load(open(src_path, "rb"))

        self.train_model()

        self.all_lfs = prune_lfs()

    def train_model(self):
        self.label_model = LabelModel(cardinality=2, verbose=True)

        labels = np.concatenate([self.simp_labels, self.src_labels])
        self.label_model.fit(L_train=labels, n_epochs=100, log_freq=5, seed=42, lr=0.001)

    def predict(self, bin_vec):
        label_model_pred_prob = self.label_model.predict_proba(L=bin_vec)[0][1]
        return np.round(label_model_pred_prob, 3)

    def transform_to_bin_vec(self, text):
        
        text = pd.DataFrame({'simplified_snt': [text]})
        applier = PandasLFApplier(self.all_lfs)
        labels = applier.apply(text, progress_bar=False)
        return labels

    def calc_score(self, text):
        bin_vec = self.transform_to_bin_vec(text)
        pred = self.predict(bin_vec)
        return pred
    
    def set_norm_weights(self):
        weights = self.label_model.get_weights()
        self.norm_weights = weights/np.sum(weights)

    def get_lfs(self):
        return self.all_lfs