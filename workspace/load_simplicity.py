import pandas as pd
import glob
import pickle


path_to_datasets = '/workspace/datasets/'

def load_asset_ds():
    # ASSET DATASET
    # https://github.com/facebookresearch/asset

    asset_path = path_to_datasets + 'asset'
    asset_df = pd.read_csv(asset_path + '/asset/human_ratings/human_ratings.csv', encoding='latin1', header=0)
        
    src_simp_simplicity = {}
    src_simp_meaning = {}
    for index, row in asset_df.iterrows():
        if row['aspect'] == 'simplicity':
            comb = row['original'] + '$$$$$' + row['simplification']
            if comb not in src_simp_simplicity:
                src_simp_simplicity[comb] = []
            
            src_simp_simplicity[comb].append(row['rating'])    
        if row['aspect'] == 'meaning':
            comb = row['original'] + '$$$$$' + row['simplification']
            if comb not in src_simp_meaning:
                src_simp_meaning[comb] = []
            
            src_simp_meaning[comb].append(row['rating'])    

    src = []
    simp = []
    simplicity = []
    meaning = []

    for key in src_simp_simplicity.keys():
        if key in src_simp_meaning:
            pts = key.split('$$$$$')
            src.append(pts[0])
            simp.append(pts[1])
            simplicity.append(sum(src_simp_simplicity[key])/len(src_simp_simplicity[key]))
            meaning.append(sum(src_simp_meaning[key])/len(src_simp_meaning[key]))

    full_data = {'ds_id': 'ASSET', 'src': src, 'simp': simp, 'simplicityScore': simplicity, 'meaningScore': meaning, 'origin': 'human'}
    asset_dataset = pd.DataFrame(data = full_data)

    with open('/' + asset_path + '/asset_simp.pkl', 'wb') as f:
        pickle.dump(asset_dataset, f)
    
    return asset_dataset

def load_metaeval_ds():
  # metaeval DATASET
  # https://github.com/feralvam/metaeval-simplification

    metaeval_path = path_to_datasets + 'metaeval'
    metaeval_df = pd.read_csv(metaeval_path + '/metaeval-simplification/data/simplicity_DA.csv', encoding='latin1', header=0)

    src = []
    simp = []
    simplicity = []
    meaning = []
    origin = []

    for index, row in metaeval_df.iterrows():   
        src.append(row['orig_sent'])
        simp.append(row['simp_sent'])
        simplicity.append(row['simplicity'])
        meaning.append(row['meaning'])
        origin.append(row['sys_name'])

    full_data = {'ds_id': 'metaeval', 'src': src, 'simp': simp, 'simplicityScore': simplicity, 'meaningScore': meaning, 'origin': origin}
    metaeval_dataset = pd.DataFrame(data = full_data)

    with open('/' + metaeval_path + '/metaeval_simp.pkl', 'wb') as f:
        pickle.dump(metaeval_dataset, f)
    
    return metaeval_dataset

def load_questeval_ds():
    # QuestEval dataset
    # http://dl.fbaipublicfiles.com/questeval/simplification_human_evaluations.tar.gz
    questeval_path = path_to_datasets + 'questeval'

    src = []
    simp = []

    questeval_df = pd.read_csv(questeval_path + '/simplification_human_evaluations/questeval_simplification_likert_ratings.csv', encoding='latin1', header=0)
    
    src_simp_simplicity = {}
    src_simp_meaning = {}
    simp_type = {}
    
    for index, row in questeval_df.iterrows():
        comb = row['source'] + '$$$$$' + row['simplification']

        if comb not in simp_type:
            simp_type[comb] = set()
        
        simp_type[comb].add(row['simplification_type'])

        if row['aspect'] == 'simplicity':
            if comb not in src_simp_simplicity:
                src_simp_simplicity[comb] = []

            src_simp_simplicity[comb].append(row['rating'])    

        if row['aspect'] == 'meaning':
            if comb not in src_simp_meaning:
                src_simp_meaning[comb] = []
            
            src_simp_meaning[comb].append(row['rating'])    
        
    src = []
    simp = []
    meaning = []
    simplicity = []
    origin = []

    for key in src_simp_simplicity.keys():
        if key in src_simp_meaning:
            pts = key.split('$$$$$')
            src.append(pts[0])
            simp.append(pts[1])
            simplicity.append(sum(src_simp_simplicity[key])/len(src_simp_simplicity[key]))
            meaning.append(sum(src_simp_meaning[key])/len(src_simp_meaning[key]))
            if len(simp_type[key]) == 1:
                origin.append(list(simp_type[key])[0])

    full_data = {'ds_id': 'QuestEval', 'src': src, 'simp': simp, 'simplicityScore': simplicity, 'meaningScore': meaning, 'origin': origin}
    questeval_dataset = pd.DataFrame(data = full_data)

    with open(questeval_path + '/questeval_mp.pkl', 'wb') as f:
        pickle.dump(questeval_dataset, f)
  
    return questeval_dataset   


def load_simpeval_ds():
  # SimpEval 2022 dataset
  # https://github.com/Yao-Dou/LENS/

    simpeval_path = path_to_datasets + 'simpeval'
    simpeval_files = sorted(glob.glob(f"{simpeval_path}/LENS/data/*"))

    src_simp_simplicity = {}
    src_simp_meaning = {}
    ds_orig = {}

    # only the non-simpeval-part of the dataset contains meaning preservation ratings, 
    # the other part does only contain ratings on the simplification
    for f in ['simpDA_2022.csv', 'simplikert_2022.csv']:
        simpeval_df = pd.read_csv(simpeval_path + '/LENS/data/' + f, encoding='latin1', header=0)

        input_col = 'Input.original'
        simplified_col = 'Input.simplified'
        
        for index, row in simpeval_df.iterrows():
            
            comb = str(row[input_col]) + '$$$$$' + str(row[simplified_col]) + '$$$$$' + str(row['Input.system']) 
            if comb not in src_simp_simplicity:
                src_simp_meaning[comb] = []
                src_simp_simplicity[comb] = []

            if f == 'simpDA_2022.csv':
                src_simp_meaning[comb].append(row['Answer.adequacy'])
                src_simp_simplicity[comb].append(row['Answer.simplicity'])
            else:
                src_simp_meaning[comb].append((row['Answer.adequacy'] - 1) * 25) # 5 point likert scale from 1 to 5 -> 0...100
                src_simp_simplicity[comb].append((row['Answer.simplicity'] + 2) * 25) # 5 point likert scale from -2 to 2 -> 0...100

            if comb not in ds_orig:
                ds_orig[comb] = [f[:-4]]
            else:
                if f[:-4] not in ds_orig[comb]:
                    ds_orig[comb].append(f[:-4])

    for f in ['simpeval_2022.csv', 'simpeval_past.csv']:
        simpeval_df = pd.read_csv(simpeval_path + '/LENS/data/' + f, encoding='latin1', header=0)

        input_col = 'original'

        if 'processed_generation' in simpeval_df.columns:
            simplified_col = 'processed_generation'
        else:
            simplified_col = 'generation'

        for index, row in simpeval_df.iterrows():
            
            comb = str(row[input_col]) + '$$$$$' + str(row[simplified_col]) + '$$$$$' + str(row['system']) 
            if comb not in src_simp_simplicity:
                src_simp_simplicity[comb] = []

            for col in simpeval_df.columns:
                if len(col) == 8 and col[:7] == 'rating_':
                    src_simp_simplicity[comb].append(row[col])

            if comb not in ds_orig:
                ds_orig[comb] = [f[:-4]]
            else:
                if f[:-4] not in ds_orig[comb]:
                    ds_orig[comb].append(f[:-4])

    src = []
    simp = []
    meaning = []
    simplicity = []
    origin = []
    inner_ds = []

    for key in src_simp_simplicity.keys():
        pts = key.split('$$$$$')
        src.append(pts[0])
        simp.append(pts[1])
        origin.append(pts[2])
        inner_ds.append(ds_orig[key])
        if key in src_simp_meaning:
            meaning.append(sum(src_simp_meaning[key])/len(src_simp_meaning[key]))
        else:
            meaning.append(-1) # means data point does not have a meaning value
        simplicity.append(sum(src_simp_simplicity[key])/len(src_simp_simplicity[key]))

    full_data = {'ds_id': 'SimpEval_22', 'src': src, 'simp': simp, 'simplicityScore': simplicity, 'meaningScore': meaning, 'origin': origin, 'inner_ds': inner_ds}
    simpeval_dataset = pd.DataFrame(data = full_data)

    with open('/' + simpeval_path + '/simpeval_mp.pkl', 'wb') as f:
        pickle.dump(simpeval_dataset, f)
    
    return simpeval_dataset


