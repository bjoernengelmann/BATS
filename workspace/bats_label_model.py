import pickle
from snorkel.labeling.model import LabelModel
import numpy as np
from snorkel.labeling import PandasLFApplier
from labeling_functions import get_all_lfs
import pandas as pd
import glob

from sklearn.utils import shuffle
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold

class BatsModel:

    @classmethod
    def get_avail_ds(cls):
        ds = set([ds.split("/")[-1].split("_")[0] for ds in sorted(glob.glob("/workspace/datasets/__all_LFs/*")) if "labels" in ds])
        ds.add("eval")
        return ds

    @classmethod
    def get_avail_tas(cls):
        ta_map = pickle.load(open("/workspace/datasets/ta_map.pkl", "rb"))
        tas = []
        avail_ds = cls.get_avail_ds()
        
        for ds in avail_ds:
            if ds in ta_map.keys():
                tas += ta_map[ds].split(",")

        return set(tas), ta_map

    @classmethod
    def get_avail_domains(cls):
        d_map = pickle.load(open("/workspace/datasets/domain_map.pkl", "rb"))
        domains = []
        avail_ds = cls.get_avail_ds()
        
        for ds in avail_ds:
            if ds in d_map.keys():
                domains += d_map[ds].split(",")

        return set(domains), d_map

    def __init__(self, ds_name):

        avail_ta, ta_map = self.get_avail_tas()
        avail_domains, domain_map = self.get_avail_domains()
        avail_ds = self.get_avail_ds()
        self.ds_name = ds_name

        if self.ds_name in avail_ta:
            #load all datasets for specified target audience

            simp_labels = []
            src_labels = []
            
            sel_ds = {k for k,v in ta_map.items() if self.ds_name in v}.intersection(avail_ds)
            
            for ds in sel_ds:
                print(f"Include: {ds}")
                simp_path = f"/workspace/datasets/__all_LFs/{ds}_simp_labels.pkl"        
                src_path = f"/workspace/datasets/__all_LFs/{ds}_src_labels.pkl"

                simp_labels.append(pickle.load(open(simp_path, "rb")))
                src_labels.append(pickle.load(open(src_path, "rb")))

            self.simp_labels = np.concatenate(simp_labels)
            self.src_labels = np.concatenate(src_labels)

        elif self.ds_name in avail_domains:
            #load all datasets for specified domain

            simp_labels = []
            src_labels = []
            
            sel_ds = {k for k,v in domain_map.items() if self.ds_name in v}.intersection(avail_ds)
            
            for ds in sel_ds:
                print(f"Include: {ds}")
                simp_path = f"/workspace/datasets/__all_LFs/{ds}_simp_labels.pkl"        
                src_path = f"/workspace/datasets/__all_LFs/{ds}_src_labels.pkl"

                simp_labels.append(pickle.load(open(simp_path, "rb")))
                src_labels.append(pickle.load(open(src_path, "rb")))

            self.simp_labels = np.concatenate(simp_labels)
            self.src_labels = np.concatenate(src_labels)

        elif self.ds_name in avail_ds:
            simp_path = f"/workspace/datasets/__all_LFs/{self.ds_name}_simp_labels.pkl"        
            src_path = f"/workspace/datasets/__all_LFs/{self.ds_name}_src_labels.pkl"  

            self.simp_labels = pickle.load(open(simp_path, "rb"))
            self.src_labels = pickle.load(open(src_path, "rb"))

        else:
            print("Please specify a correct dataset/target audience/domain")
            print("Following are available:")
            print(f"Datasets: {avail_ds}")
            print(f"Target Audience: {avail_ta}")
            print(f"Domains: {avail_domains}")

            return 

        self.train_model()

        self.all_lfs = get_all_lfs()
        self.set_norm_weights()

        self.classifier = None


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

    def transform_to_bin_vec_evol(self, text):
        
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

    def calc_naive_score(self, text):
        weights = self.norm_weights
        t1_bin = self.transform_to_bin_vec(text)
        
        s_val = 0
        ns_val = 0

        for i in range(len(t1_bin[0])):
            if t1_bin[0][i] == 0:
                s_val += weights[i]
            elif t1_bin[0][i] == 1:
                ns_val += weights[i]

        return ns_val / (s_val + ns_val)

    def get_opt_lfs():
        return self.lfs_vec

    def get_classifier(self):
        if self.classifier:
            return self.classifier
        
        path_to_lf = f"/workspace/datasets/gen_opt_lfs/{self.ds_name}_lfs.pkl"
        self.lfs_vec = pickle.load(open(path_to_lf, "rb"))


        #train classifier
        X, y = np.concatenate([self.simp_labels, self.src_labels]), np.array([0]*len(self.simp_labels) + [1]*len(self.src_labels))
        X, y = shuffle(X, y, random_state=42)

        #only use optimal LFs
        sel_indices = np.array(self.lfs_vec).nonzero()[0]
        opt_X = np.take(X, sel_indices, axis=1)

        self.classifier = RandomForestClassifier(random_state=42)
        self.classifier.fit(opt_X, y)

        return self.classifier

    def classify(self, bin_vec):

        _ = self.get_classifier()

        if bin_vec.shape[0] == self.lfs_vec.shape[0]:
             #opt lfs transformtation
            sel_indices = np.array(self.lfs_vec).nonzero()[0]
            opt_X = np.take(bin_vec, sel_indices)
        elif bin_vec.shape[0] == np.sum(self.lfs_vec):
            opt_X = bin_vec

        elif bin_vec.shape[0] * 2 ==  self.lfs_vec.shape[0]:
            #input is not categorical
            raise NotImplementedError

        else:
            print("bad input")
            return

        return self.classifier.predict(opt_X.reshape(1,-1))

