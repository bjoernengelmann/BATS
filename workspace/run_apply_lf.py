import argparse
import pickle
import os
from pruning_lfs import prune_lfs
import glob
from tqdm import tqdm
import pandas as pd

from snorkel.labeling import PandasLFApplier


def preprocess_dataset(dataset, sel_ds_id, app_type):
    dataset = dataset[dataset['ds_id'] == sel_ds_id]
    dataset['simplified_snt'] = dataset[app_type]
    dataset['source_snt'] = dataset['src']

    return dataset

def get_finished_batches(sel_ds_id, app_type):
    path = f"datasets/{sel_ds_id}labels/*"
    label_paths = glob.glob(path)
    fin_batches = [int(path.split("_")[-1]) for path in label_paths if "labels_" in path and app_type in path]
    return set(fin_batches)

def save_used_lfs(all_lfs, sel_ds_id):
    path = f"datasets/{sel_ds_id}labels/used_lfs.pkl"
    names = [lf.name for lf in all_lfs]
    pickle.dump(names, open(path, "wb"))


if __name__ == "__main__":

    all_lfs = prune_lfs()

    print("loading dataset file ...")
    with open("datasets/final_combined_with_index.pkl", 'rb') as f:
        dataset = pickle.load(f)
    
    print("dataset loaded")
    ds_ids = dataset['ds_id'].unique().tolist()

    parser = argparse.ArgumentParser(description="Run pruned lfs on specified dataset")
    parser.add_argument('ds', type=str, help=f"Please specify one of the following datasets: {ds_ids}")
    parser.add_argument('apply_on', type=str, help=f"apply lfs on source or simplified texts")
    parser.add_argument('--batch_size', type=int, help="number of texts which are processed at once")

    args = parser.parse_args()

    sel_ds_id = args.ds
    app_type = args.apply_on

    if not sel_ds_id in ds_ids:
        raise Exception(f"{sel_ds_id} is not available, choose one of the following: \n {ds_ids}")

    if not app_type in ['src', 'simp']:
        raise Exception(f"{app_type} is not available, choose one of the following: \n {['src', 'simp']}")

    label_path = f"datasets/{sel_ds_id}labels/"

    dataset = preprocess_dataset(dataset, sel_ds_id, app_type)

    if not os.path.isdir(label_path):
        os.mkdir(label_path)
    

    batch_size = 20
    start = 0

    save_used_lfs(all_lfs, sel_ds_id)
    print()
    

    for i in tqdm(range(start, len(dataset), batch_size), position=1):
        if not i in get_finished_batches(sel_ds_id, app_type):
            try:
                applier = PandasLFApplier(all_lfs)
                labels = applier.apply(dataset[i:i+batch_size], progress_bar=True)
                
                pickle.dump(labels, open(f"datasets/{sel_ds_id}labels/labels_{app_type}_{i}", "wb"))
                print(f"finished on {i}/{len(dataset)}")

            except Exception as e:
                print(f"something went wrong with batch {i}")
                print(e)

    print("Successfully finished")
