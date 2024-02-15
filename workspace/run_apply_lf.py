import argparse
import pickle
import os
from labeling_functions import get_all_lfs

import glob
from tqdm import tqdm
import pandas as pd
from multiprocessing import Process

from snorkel.labeling import PandasLFApplier


def preprocess_dataset(dataset, sel_ds_id, app_type):
    dataset = dataset[dataset['ds_id'] == sel_ds_id]        

    dataset['simplified_snt'] = dataset[app_type]
    dataset['source_snt'] = dataset['src']

    return dataset

def get_finished_batches(sel_ds_id, app_type, all_lfs):
    path = f"datasets/{sel_ds_id}_{len(all_lfs)}_parts/*"
    label_paths = glob.glob(path)
    fin_batches = [int(path.split("_")[-1]) for path in label_paths if "labels_" in path and app_type in path]
    return set(fin_batches)

def save_used_lfs(all_lfs, sel_ds_id):
    path = f"datasets/{sel_ds_id}_{len(all_lfs)}_parts/used_lfs.pkl"
    names = [lf.name for lf in all_lfs]
    pickle.dump(names, open(path, "wb"))


def apply_lfs(df, lfs, chunk_start, size, sel_ds_id, app_type):
    applier = PandasLFApplier(lfs)
    labels = applier.apply(df[chunk_start:chunk_start+size], progress_bar=False)
    pickle.dump(labels, open(f"datasets/{sel_ds_id}_{len(all_lfs)}_parts/labels_{app_type}_{chunk_start}", "wb"))


if __name__ == "__main__":

    print("loading dataset file ...")
    with open("datasets/final_combined_with_index.pkl", 'rb') as f:
        dataset = pickle.load(f)
    
    print("dataset loaded")
    ds_ids = dataset['ds_id'].unique().tolist()
    ds_ids.append("eval")

    parser = argparse.ArgumentParser(description="Run pruned lfs on specified dataset")
    parser.add_argument('ds', type=str, help=f"Please specify one of the following datasets: {ds_ids}")
    parser.add_argument('apply_on', type=str, help=f"apply lfs on source or simplified texts")
    parser.add_argument('full', type=bool, help="wheter to apply pruned set of LF or all")
    parser.add_argument('--batch_size', type=int, help="number of texts which are processed at once")
    parser.add_argument('--parallel', type=int, help="number of cores to use for parallelisation (should be divisor of batch_size)")


    args = parser.parse_args()
    
    batch_size = 50
    n_parallel = 5

    batch_size = args.batch_size
    n_parallel = args.parallel
    sel_ds_id = args.ds
    app_type = args.apply_on
    full = args.full

    if not sel_ds_id in ds_ids:
        raise Exception(f"{sel_ds_id} is not available, choose one of the following: \n {ds_ids}")

    if not app_type in ['src', 'simp']:
        raise Exception(f"{app_type} is not available, choose one of the following: \n {['src', 'simp']}")

    all_lfs = get_all_lfs()

    label_path = f"datasets/{sel_ds_id}_{len(all_lfs)}_parts/"

    dataset = preprocess_dataset(dataset, sel_ds_id, app_type)

    if not os.path.isdir(label_path):
        os.mkdir(label_path)
    

    

    

    start = 0

    save_used_lfs(all_lfs, sel_ds_id)
    print()
    

    for i in tqdm(range(start, len(dataset), batch_size), position=1):
        if not i in get_finished_batches(sel_ds_id, app_type, all_lfs):
            try:
                
                procs = []
                for j in range(n_parallel):
                    
                    chunk_size = int(batch_size/n_parallel)
                    chunk_start = i + j*chunk_size

                    if chunk_start > len(dataset):
                        continue

                    proc = Process(target=apply_lfs, args=(dataset, all_lfs, chunk_start, chunk_size, sel_ds_id, app_type))
                    procs.append(proc)
                    proc.start()

                for proc in procs:
                    proc.join()
             
                print(f"finished on {i}/{len(dataset)}")

            except Exception as e:
                print(f"something went wrong with batch {i}")
                print(e)

    print("Successfully finished")
