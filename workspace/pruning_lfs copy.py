import pickle
import numpy as np
from labeling_functions import get_all_lfs
import pandas as pd
import os
import tqdm
from tqdm import tqdm

from functools import lru_cache

from snorkel.labeling import labeling_function
from snorkel.labeling import LabelingFunction
from snorkel.labeling.model import LabelModel
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LFAnalysis

import warnings


def prune_lfs():
    all_lfs = get_all_lfs()
    labels_simp = pickle.load(open("/workspace/datasets/eval_simp_labels.pkl", "rb"))
    labels_src = pickle.load(open("/workspace/datasets/eval_src_labels.pkl", "rb"))

    lfa_simp = LFAnalysis(L=labels_simp, lfs=all_lfs).lf_summary()
    lfa_src = LFAnalysis(L=labels_src, lfs=all_lfs).lf_summary()
    
    df_simp = pd.DataFrame(lfa_simp)
    df_src = pd.DataFrame(lfa_src)  

    merged_data = []

    for index, row in df_simp.iterrows():
        polarity = -1
        cov_simp = row['Coverage']
        cov_src = df_src.loc[index]['Coverage']

        if len(row['Polarity']) > 0:
            polarity = row['Polarity'][0]
        else:
            if len(df_src.loc[index]['Polarity']) > 0:
                polarity = df_src.loc[index]['Polarity'][0]
                
        precision = 0

        if polarity == 0:
            precision = cov_simp/(cov_simp + cov_src)
        else:
            if polarity == 1:
                precision = cov_src/(cov_simp + cov_src)

        merged_data.append([index, polarity, cov_simp, cov_src, precision, 1 - precision, cov_simp+cov_src, abs(cov_simp - cov_src), abs(cov_simp - cov_src)/(cov_simp+cov_src)])

    df_md = pd.DataFrame(merged_data)
    df_md.columns = ['name', 'polarity', 'cov_simp', 'cov_src', 'precision', 'inv_precision', 'total_coverage', 'distance', 'norm_dist']

    decision = []

    for index, row in df_md.iterrows():
        if row['precision'] > 0:
            if row['total_coverage'] >= 0.05:
                if row['precision'] >= 0.7:
                    decision.append('JA')
                else: 
                    if row['precision'] >= 0.5 and row['norm_dist'] >= 0.05:
                        decision.append('JA')
                    else:
                        decision.append('NEIN')
            else:
                if row['precision'] > 0.5 and row['distance'] >= 0.005 and row['total_coverage'] >= 0.02:
                    decision.append('JA')
                else:
                    decision.append('NEIN')
        else:
            decision.append('NEIN')

    df_md['decision'] = decision

    df_md.to_excel("/workspace/datasets/merged_label_data.xlsx")  

    keepLFs = []

    for index, row in df_md.iterrows():
        if row['decision'] == 'JA':
            keepLFs.append(row['name'])

    selected_lfs = [a for a in all_lfs if a.name in keepLFs]

    return selected_lfs
