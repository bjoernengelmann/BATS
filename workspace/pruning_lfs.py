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

#datasets = []
def prune_lfs(CHOSEN_DS = '', TARGETAUDIENCE = None, DOMAIN = None):
    all_lfs = get_all_lfs()
    datasets = []

    if len(CHOSEN_DS) > 0:
        datasets = [CHOSEN_DS]
    else:
        metadata_ds = pd.read_excel('/workspace/datasets/English_Datasets.xlsx')

        for index, row in metadata_ds.iterrows():
            print(row['Target_Audience'])
            if TARGETAUDIENCE in row['Target_Audience'].split(', ') or DOMAIN in row['Domain'].split(', '):
                datasets.append(row['ds_id'])

    labels_simp_list = []
    labels_src_list = []
    for dataset in datasets:
        labels_simp_list.append(pickle.load(open("/workspace/datasets/__all_LFs/" + dataset + "_simp_labels.pkl", "rb")))
        labels_src_list.append(pickle.load(open("/workspace/datasets/__all_LFs/" + dataset + "_src_labels.pkl", "rb")))

    lfa_simp = LFAnalysis(L=np.vstack(labels_simp_list), lfs=all_lfs).lf_summary()
    lfa_src = LFAnalysis(L=np.vstack(labels_src_list), lfs=all_lfs).lf_summary()
    
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
                    decision.append('YES')
                else: 
                    if row['precision'] >= 0.5 and row['norm_dist'] >= 0.05:
                        decision.append('YES')
                    else:
                        decision.append('NO')
            else:
                if row['precision'] > 0.5 and row['distance'] >= 0.005 and row['total_coverage'] >= 0.02:
                    decision.append('YES')
                else:
                    decision.append('NO')
        else:
            decision.append('NO')

    df_md['decision'] = decision

    df_md.to_excel("/workspace/datasets/__all_LFs/prunings/" + str(datasets) + "_" + str(TARGETAUDIENCE) + "_" + str(DOMAIN) + "_label_data.xlsx")  

    keepLFs = []

    for index, row in df_md.iterrows():
        if row['decision'] == 'YES':
            keepLFs.append(row['name'])

    selected_lfs = [a for a in all_lfs if a.name in keepLFs]

    return selected_lfs
