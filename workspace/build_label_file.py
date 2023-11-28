import glob
import argparse
import pickle
import numpy as np

if __name__== "__main__":
    parser = argparse.ArgumentParser(description="Run build label file on specified dataset")
    parser.add_argument('ds', type=str, help=f"Please specify ds")
    parser.add_argument('num_lfs', type=int, help=f"Please specify num of used LF")

    args = parser.parse_args()

    ds_label_path = "datasets/ds_labels"

    sel_ds_id = args.ds
    num_lfs = args.num_lfs

    src_part_paths = sorted(glob.glob(f"datasets/{sel_ds_id}_{num_lfs}_parts/labels_src*"), key=lambda p: int(p.split('_')[-1]))
    simp_part_paths = sorted(glob.glob(f"datasets/{sel_ds_id}_{num_lfs}_parts/labels_simp*"), key=lambda p: int(p.split('_')[-1]))

    src_labels = [pickle.load(open(path, "rb")) for path in src_part_paths]
    simp_labels = [pickle.load(open(path, "rb")) for path in simp_part_paths]
    
    src_labels = np.concatenate(src_labels)
    simp_labels = np.concatenate(simp_labels)
    
    pickle.dump(src_labels, open(f"{ds_label_path}/{sel_ds_id}-{num_lfs}_src_labels.pkl", "wb"))
    pickle.dump(simp_labels, open(f"{ds_label_path}/{sel_ds_id}-{num_lfs}_simp_labels.pkl", "wb"))
    