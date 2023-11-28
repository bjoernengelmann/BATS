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

    src = []
    simp = []
    system = []
    ds_orig = []
    simplicity = []
    references = []


    # wikiDA, 600 entries

    # references for wikiDA
    refs = {}

    with open(simpeval_path + '/LENS/experiments/meta_evaluation/data/wikida/references/test.src', encoding='latin1') as f:
        src_lines = f.readlines()

        with open(simpeval_path + '/LENS/experiments/meta_evaluation/data/wikida/references/test.dst', encoding='latin1') as f2:
            ref_lines = f2.readlines()

            for i in range(0, len(src_lines)):
                refs[src_lines[i].strip()] = ref_lines[i].strip().split('\t')
    
    simpeval_df = pd.read_csv(simpeval_path + '/LENS/experiments/meta_evaluation/data/wikida/simplicity_DA.csv', encoding='latin1', header=0)

    for index, row in simpeval_df.iterrows():
        src.append(row['orig_sent'])
        simp.append(row['simp_sent'])
        system.append(row['sys_type'])
        ds_orig.append('simplicity_DA')

        simplicity.append(row['simplicity'])
        references.append(refs[row['orig_sent'].strip()])

    # simpeval_2022, 360 entries

    # references for simpeval_2022
    refs = {}

    with open(simpeval_path + '/LENS/experiments/meta_evaluation/data/simpeval_2022/references/test.src', encoding='latin1') as f:
        src_lines = f.readlines()

        with open(simpeval_path + '/LENS/experiments/meta_evaluation/data/simpeval_2022/references/test.dst', encoding='latin1') as f2:
            ref_lines = f2.readlines()

            for i in range(0, len(src_lines)):
                refs[src_lines[i].strip()] = [ref_lines[i].strip()]

    simpeval_df = pd.read_csv(simpeval_path + '/LENS/experiments/meta_evaluation/data/simpeval_2022/simpeval_2022.csv', encoding='latin1', header=0)

    for index, row in simpeval_df.iterrows():
        src.append(row['original'])
        simp.append(row['generation'])
        system.append(row['system'])
        ds_orig.append('simpeval_2022')

        simplicity.append(row['average'])
        references.append(refs[row['original'].strip()])

    full_data = {'ds_id': 'SimpEval_22', 'src': src, 'simp': simp, 'simplicityScore': simplicity, 'origin': system, 'inner_ds': ds_orig, 'references': references}
    simpeval_dataset = pd.DataFrame(data = full_data)

    with open('/' + simpeval_path + '/simpeval_mp.pkl', 'wb') as f:
        pickle.dump(simpeval_dataset, f)
    
    return simpeval_dataset