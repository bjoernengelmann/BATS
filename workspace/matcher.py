import pandas as pd
import re
from labeling_functions import get_all_lfs

def get_mapping_of_LF_and_dims():
    manualLF2C = pd.read_excel("/workspace/datasets/__clustering_of_LFs/manualLF2C.xlsx")
    all_lfs = get_all_lfs()
    names = [a.name for a in all_lfs]
    dims_to_names = {}
    names_to_dims = {}

    for a in names:
        dims_to_names[len(dims_to_names)] = [a]
        names_to_dims[a] = len(names_to_dims)

    animals = ["dog", "fish", "sheep", "bunny", "octopus", "roadrunner", "okapi", "anisakis", "bat", "leopard" ]

    thresless_LFs = {}

    for index, row in manualLF2C.iterrows():
        curr_meth = row['Methode']
        pts_left_brackets = curr_meth.split('{')

        cryptic_name = pts_left_brackets[0]
        for pt in pts_left_brackets:
            pts_right_brackets = pt.split('}')

            if len(pts_right_brackets) > 1:
                cryptic_name = cryptic_name + 'XXX' + pts_right_brackets[1]
        
        thresless_LFs[cryptic_name] = curr_meth

    LFs_to_thresless = {}
    LFs_to_cat = {}
    LF_to_feature = {}


    for lf in names:
        curr_lf = lf.replace('label=1', 'label=NOT_SIMPLE').replace('label=0', 'label=SIMPLE')

        if 'label' not in curr_lf and 'SIMPLE' not in curr_lf:
            if curr_lf[-2:] == '_1':
                curr_lf = curr_lf[:-2] + '_NOT_SIMPLE'
            if curr_lf[-2:] == '_0':
                curr_lf = curr_lf[:-2] + '_SIMPLE'
            
        wo_percentage = re.sub('\d+\.\d+', 'XXX', curr_lf)
        curr_lf = re.sub('\d+', 'XXX', wo_percentage)
        
        for to_exclude in animals:
            curr_lf = curr_lf.replace('_' + to_exclude + '_', '_XXX_')

        if 'perc_more_than_XXX_characters' in curr_lf:
            curr_lf = curr_lf.replace('perc_more_than_XXX_characters', 'perc_more_than_8_characters')
        
        if curr_lf not in thresless_LFs.keys():
            print(curr_lf)
        
        LFs_to_thresless[lf] = manualLF2C[manualLF2C['Methode'] == thresless_LFs[curr_lf]]['clustered'].values[0]#['cluster'].values[0]
        LFs_to_cat[lf] = manualLF2C[manualLF2C['Methode'] == thresless_LFs[curr_lf]]['cluster'].values[0]
        LF_to_feature[lf] = manualLF2C[manualLF2C['Methode'] == thresless_LFs[curr_lf]]['Lit_short'].values[0]



    #overview = {} # how often do we find a LF belonging to Structural, Lexical, Pragmatic, Syntactic
    #labels_to_clusters = {}
    #final_clustering_of_manual_LFs = {} # use this to compare to automatically clustered LFs

    #for lf in LFs_to_thresless.keys():
    #    if LFs_to_thresless[lf] in overview:
    #        overview[LFs_to_thresless[lf]] = overview[LFs_to_thresless[lf]] + 1
    #    else:
    #        labels_to_clusters[LFs_to_thresless[lf]] = len(labels_to_clusters)
    #        overview[LFs_to_thresless[lf]] = 1

    #    final_clustering_of_manual_LFs[lf] = labels_to_clusters[LFs_to_thresless[lf]]

    return dims_to_names, names_to_dims, LFs_to_thresless, LFs_to_cat, LF_to_feature