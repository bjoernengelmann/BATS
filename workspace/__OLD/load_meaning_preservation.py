import pandas as pd
import glob
import pickle


path_to_datasets = '/workspace/datasets/'

def load_asset_ds():
    # ASSET DATASET
    # https://github.com/facebookresearch/asset

    asset_path = path_to_datasets + 'asset'
    asset_df = pd.read_csv(asset_path + '/asset/human_ratings/human_ratings.csv', encoding='latin1', header=0)
        
    src_simp = {}
    for index, row in asset_df.iterrows():
        if row['aspect'] == 'meaning':
            comb = row['original'] + '$$$$$' + row['simplification']
            if comb not in src_simp:
                src_simp[comb] = []
            
            src_simp[comb].append(row['rating'])    

    src = []
    simp = []
    meaning = []

    for key in src_simp.keys():
        pts = key.split('$$$$$')
        src.append(pts[0])
        simp.append(pts[1])
        meaning.append(sum(src_simp[key])/len(src_simp[key]))

    full_data = {'ds_id': 'ASSET', 'src': src, 'simp': simp, 'meaningScore': meaning}
    asset_dataset = pd.DataFrame(data = full_data)

    with open('/' + asset_path + '/asset_mp.pkl', 'wb') as f:
        pickle.dump(asset_dataset, f)
    
    return asset_dataset

def load_metaeval_ds():
  # metaeval DATASET
  # https://github.com/feralvam/metaeval-simplification

    metaeval_path = path_to_datasets + 'metaeval'
    metaeval_df = pd.read_csv(metaeval_path + '/metaeval-simplification/data/simplicity_DA.csv', encoding='latin1', header=0)

    src = []
    simp = []
    meaning = []

    for index, row in metaeval_df.iterrows():   
        src.append(row['orig_sent'])
        simp.append(row['simp_sent'])
        meaning.append(row['meaning'])

    full_data = {'ds_id': 'metaeval', 'src': src, 'simp': simp, 'meaningScore': meaning}
    metaeval_dataset = pd.DataFrame(data = full_data)

    with open('/' + metaeval_path + '/metaeval_mp.pkl', 'wb') as f:
        pickle.dump(metaeval_dataset, f)
    
    return metaeval_dataset

def load_questeval_ds():
    # QuestEval dataset
    # http://dl.fbaipublicfiles.com/questeval/simplification_human_evaluations.tar.gz
    questeval_path = path_to_datasets + 'questeval'

    src = []
    simp = []

    questeval_df = pd.read_csv(questeval_path + '/simplification_human_evaluations/questeval_simplification_likert_ratings.csv', encoding='latin1', header=0)
    
    src_simp = {}
    for index, row in questeval_df.iterrows():
        if row['aspect'] == 'meaning':
            comb = row['source'] + '$$$$$' + row['simplification']
            if comb not in src_simp:
                src_simp[comb] = []
            
            src_simp[comb].append(row['rating'])    

    src = []
    simp = []
    meaning = []

    for key in src_simp.keys():
        pts = key.split('$$$$$')
        src.append(pts[0])
        simp.append(pts[1])
        meaning.append(sum(src_simp[key])/len(src_simp[key]))


    full_data = {'ds_id': 'QuestEval', 'src': src, 'simp': simp, 'meaningScore': meaning}
    questeval_dataset = pd.DataFrame(data = full_data)

    with open(questeval_path + '/questeval_mp.pkl', 'wb') as f:
        pickle.dump(questeval_dataset, f)
  
    return questeval_dataset   


def load_simpeval_ds():
  # SimpEval 2022 dataset
  # https://github.com/Yao-Dou/LENS/

    simpeval_path = path_to_datasets + 'simpeval'
    src_simp = {}

    # only the non-simpeval-part of the dataset contains meaning preservation ratings, 
    # the other part does only contain ratings on the simplification
    for f in ['simpDA_2022.csv', 'simplikert_2022.csv']:
        simpeval_df = pd.read_csv(simpeval_path + '/LENS/data/' + f, encoding='latin1', header=0)

        input_col = 'Input.original'
        simplified_col = 'Input.simplified'
        
        for index, row in simpeval_df.iterrows():
            
            comb = str(row[input_col]) + '$$$$$' + str(row[simplified_col])
            if comb not in src_simp:
                src_simp[comb] = []
            
            if f == 'simpDA_2022.csv':
                src_simp[comb].append(row['Answer.adequacy'])
            else:
                src_simp[comb].append((row['Answer.adequacy'] - 1) * 25) # 5 point likert scale from 1 to 5 -> 0...100

    src = []
    simp = []
    meaning = []

    for key in src_simp.keys():
        if len(src_simp[key]) < 1:
            print(key)

    for key in src_simp.keys():
        pts = key.split('$$$$$')
        src.append(pts[0])
        simp.append(pts[1])
        meaning.append(sum(src_simp[key])/len(src_simp[key]))

    full_data = {'ds_id': 'SimpEval_22', 'src': src, 'simp': simp, 'meaningScore': meaning}
    simpeval_dataset = pd.DataFrame(data = full_data)

    with open('/' + simpeval_path + '/simpeval_mp.pkl', 'wb') as f:
        pickle.dump(simpeval_dataset, f)
    
    return simpeval_dataset


