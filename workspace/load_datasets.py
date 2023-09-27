import os
import pandas as pd
import glob
import numpy as np
import pickle

path_to_datasets = os.getcwd() + '/datasets/'

def load_list_file(path):
  data = []
  with open(path, "r") as f:
    for line in f.readlines():
      data.append(line.strip())
  return data

def load_asset_ds():
  # ASSET DATASET
  # https://github.com/facebookresearch/asset

  if not os.path.isfile(path_to_datasets + '/asset/asset.pkl'):
    asset_path = path_to_datasets + 'asset'
    asset_link = 'https://github.com/facebookresearch/asset'

    if not os.path.isdir(asset_path):
      os.mkdir(asset_path)
      os.chdir(asset_path)
      clone = 'git clone ' + asset_link
      os.system(clone)

    asset_files = sorted(glob.glob(f"{asset_path}/asset/dataset/*"))

    asset_test_orig = load_list_file([path for path in asset_files if ".test.orig" in path][0])
    asset_valid_orig = load_list_file([path for path in asset_files if ".valid.orig" in path][0])

    asset_test_simps = [load_list_file(path) for path in asset_files if "test.simp" in path]
    asset_valid_simps = [load_list_file(path) for path in asset_files if "valid.simp" in path]

    src = asset_test_orig * len(asset_test_simps) + asset_valid_orig * len(asset_test_simps)
    label = ['test'] * len(asset_test_orig) * len(asset_test_simps) + ['valid'] * len(asset_valid_orig) * len(asset_test_simps)

    src_ids = []
    for j in range(len(asset_test_simps)):
      for i in range(len(asset_test_orig)):
        src_ids.append(i)
    for j in range(len(asset_test_simps)):
      for i in range(len(asset_valid_orig)):
        src_ids.append(i + len(asset_test_orig))
    simp = []
    origin = []

    simp_ids = range(0, (len(asset_test_orig + asset_valid_orig)) * len(asset_test_simps)) 

    for i in range(len(asset_test_simps)):
        for j in asset_test_simps[i]:
          simp.append(j)
          origin.append('annotator_' + str(i))

    for i in range(len(asset_valid_simps)):
        for j in asset_valid_simps[i]:
          simp.append(j)
          origin.append('annotator_' + str(i))

    full_data = {'ds_id' : 'ASSET', 'src' : src, 'src_id' : src_ids, 'simp' : simp, 'simp_id' : simp_ids, 'label' : label, 'origin' : origin}

    asset_dataset = pd.DataFrame(data=full_data)

    with open(asset_path + '/asset.pkl', 'wb') as f:
      pickle.dump(asset_dataset, f)

    #todo: metadata for dataset
  else:
    asset_dataset = pd.read_pickle(path_to_datasets + '/asset/asset.pkl')

  return asset_dataset

def load_htss_ds():
  #HTSS DATASET
  #https://github.com/slab-itu/HTSS/

  htss_path_base = "/workspace/datasets/htss"
  htss_file = htss_path_base + '/new_data_set.csv'

  # htss_simp_df = pd.read_csv(htss_file, encoding='latin1', header=0)
  # htss_simp_df = htss_simp_df.reset_index()

  # htss_simp_title = []
  # htss_simp_text = []
  # htss_source_title = []
  # htss_source_text = []

  # for index, row in tqdm(htss_simp_df.iterrows(), total=df.shape[0]):
  #   source = htss_path_base + '/' + row['Full_Paper_XML'][:-3] + 'txt'

  #   if os.path.exists(source):
  #     htss_simp_title.append(row['Eureka_Title_Simplified'])
  #     htss_simp_text.append(row['Eureka_Text_Simplified'])
  #     htss_source_title.append(row['Paper_Title'])

  #     with open(source) as f:
  #       lines = f.readlines()
  #       # first four lines are ID and title of paper and can thus be ignored
  #       doc = str(lines[4:])[2:-2].replace('\\n', '').replace('", "', "', '").replace('", ', "', ").replace(', "', ", '").replace("', '", '').replace('\\r', '').replace('.  #@NEW_LINE#@#  ', '. ').replace('#@NEW_LINE#@#', '\\n').replace('  \\n  ', ' \\n ').replace('"', '').replace("'", '')
  #       htss_source_text.append(doc)

  # full_data = {'orig_snt' : htss_source_text, 'orig_snt_title' : htss_source_title, 'simp' : htss_simp_text, 'simp_snt_title' : htss_simp_title}
  # htss_dataset = pd.DataFrame(data = full_data)
  htss_dataset = pd.read_pickle(htss_path_base + '/pickled_df.pkl')
  return htss_dataset


def load_ebbe_ds():
  #EBBE dataset (authors do not actually give it a name)
  #http://www.cs.columbia.edu/~noemie/alignment/

  ebbe_path = "/workspace/datasets/EBBE"
  ebbe_files = sorted(glob.glob(f"{ebbe_path}/*"))

  ebbe_simp_text = []
  ebbe_source_text = []
  ebbe_label = []

  for path in ebbe_files:
    with open(path) as f:
      lines = f.readlines()
      for l_id in range(0, len(lines), 3):
        if len(lines[l_id].strip()) > 0 and len(lines[l_id + 1].strip()) > 0:
          ebbe_source_text.append(lines[l_id].replace(' .\\n', '.'))
          ebbe_simp_text.append(lines[l_id + 1].replace(' .\\n', '.'))
          ebbe_label.append(path[len(ebbe_path) + 1:-4])

  full_data = {'orig_snt' : ebbe_source_text, 'simp' : ebbe_simp_text, 'label' : ebbe_label}
  ebbe_dataset = pd.DataFrame(data = full_data)
  return ebbe_dataset

def load_simpa_ds():
  #SIMPA DATASET
  #https://github.com/simpaticoproject/simpa

  if not os.path.isfile(path_to_datasets + '/simpa/simpa.pkl'):
    simpa_path = path_to_datasets + 'simpa'
    simpa_link = 'https://github.com/simpaticoproject/simpa'

    if not os.path.isdir(simpa_path):
      os.mkdir(simpa_path)
      os.chdir(simpa_path)
      clone = 'git clone ' + simpa_link
      os.system(clone)
      os.remove(simpa_path + '/simpa/README.md')

    simpa_files = sorted(glob.glob(f"{simpa_path}/simpa/*"))

    simpa_ls_orig = load_list_file(simpa_files[0])
    simpa_ls_simp = load_list_file(simpa_files[1])

    simpa_ss_orig = load_list_file(simpa_files[3])
    simpa_ss_simp = load_list_file(simpa_files[4])
    simpa_ss_ls_simp = load_list_file(simpa_files[2])


    # ls_source -> ls_simp, ss_source -> ss_ll_simp, ss_source -> ss_simp, ss_ll_simp -> ss_simp
    src_ids = []
    for i in range(int(len(simpa_ls_orig)/3)):
      for j in range(3):
        src_ids.append(i)
  
    for j in range(2):
      for i in range(len(simpa_ss_orig)):
        src_ids.append(i + int(len(simpa_ls_orig)/3))

    simp_ids = []
    for i in range(len(src_ids)):
      simp_ids.append(i)

    curr_last_src_id = src_ids[-1] + 1
    curr_last_simp_id = 4400
    for i in range(len(simpa_ss_ls_simp)):
      src_ids.append(i + curr_last_src_id)
      simp_ids.append(i + curr_last_simp_id)

    src = simpa_ls_orig + simpa_ss_orig + simpa_ss_orig + simpa_ss_ls_simp
    simp = simpa_ls_simp + simpa_ss_ls_simp + simpa_ss_simp + simpa_ss_simp
    origin = ['lexical_simp'] * len(simpa_ls_orig) + ['syntactic_simp'] * len(simpa_ss_ls_simp) + ['lexical_simp_of_syntactic_simp'] * len(simpa_ss_simp) * 2

    full_data = {'ds_id' : 'simpa', 'src' : src, 'src_id' : src_ids, 'simp' : simp, 'simp_id' : simp_ids, 'label' : 'test', 'origin' : origin}

    simpa_dataset = pd.DataFrame(data = full_data)

    with open(simpa_path + '/simpa.pkl', 'wb') as f:
      pickle.dump(simpa_dataset, f)

    #todo: metadata for dataset
  else:
    simpa_dataset = pd.read_pickle(path_to_datasets + '/simpa/simpa.pkl')

  return simpa_dataset

def load_pwkp_ds():
  #PWKP dataset
  #https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2447

  pwkp_path = "/workspace/datasets/PWKP/PWKP_108016"

  pwkp_simp_text = []
  pwkp_source_text = []

  with open(pwkp_path) as f:
    lines = f.readlines()
    # first: complex sentence, then up to multiple simplified ones
    source_sent = ''
    simp_sents = ''

    last_line_blank = True
    for l in lines:
      if len(l.strip()) > 0:
        if last_line_blank:
          source_sent = l.strip()
          last_line_blank = False
        else:
          simp_sents = simp_sents + l.strip() + ' '
      else:
        if len(simp_sents) > 0 and len(source_sent) > 0:
          pwkp_simp_text.append(simp_sents)
          pwkp_source_text.append(source_sent)

        last_line_blank = True
        simp_sents = ''
        source_sent = ''

  full_data = {'orig_snt' : pwkp_source_text, 'simp' : pwkp_simp_text}
  pwkp_dataset = pd.DataFrame(data = full_data)
  return pwkp_dataset

def load_rnd_st_ds():
  df_simplified = pd.read_json("/workspace/datasets/simple_text_runfiles/irgc_task_3_ChatGPT_2stepTurbo.json")
  df_source = pd.read_json("/workspace/datasets/simple_text/simpletext-task3-test-large.json")

  sample_ids = df_simplified['snt_id'][:200]
  source_sample = df_source[df_source['snt_id'].isin(sample_ids)]

  simp_sample = df_simplified[df_simplified['snt_id'].isin(sample_ids)]

  sub_sample = source_sample.merge(simp_sample).drop_duplicates(subset=['snt_id'])

  #lets say we have some gold decisions wether a simplified version is good or not (0 means simple)
  sub_sample['gold_label'] = np.array(np.random.random(len(sub_sample))< 0.2, dtype=int)
  sub_sample = sub_sample.drop(['doc_id', 'query_id', 'query_text', 'run_id', 'manual'], axis=1)

  return sub_sample



def main():
  if not os.path.isdir(path_to_datasets):
    os.mkdir(path_to_datasets)
  ds = load_simpa_ds()

if __name__ == '__main__':
  main()
