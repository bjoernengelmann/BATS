import os
import pandas as pd
import glob
import numpy as np

def load_list_file(path):
  data = []
  with open(path, "r") as f:
    for line in f.readlines():
      data.append(line.strip())
  return data

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

  simpa_path = "/workspace/datasets/simpa"
  simpa_files = sorted(glob.glob(f"{simpa_path}/*"))

  simpa_ls_orig = load_list_file(simpa_files[0])
  simpa_ls_simp = load_list_file(simpa_files[1])

  simpa_ss_orig = load_list_file(simpa_files[3])
  simpa_ss_simp = load_list_file(simpa_files[4])
  simpa_ss_ls_simp = load_list_file(simpa_files[2])

  full_data_ls = {'orig_snt' : simpa_ls_orig, 'simp' : simpa_ls_simp}
  full_data_ss = {'orig_snt' : simpa_ss_orig, 'lex_simp' : simpa_ss_ls_simp, 'syn_simp' : simpa_ss_simp}

  simpa_dataset_ls = pd.DataFrame(data = full_data_ls)
  simpa_dataset_ss = pd.DataFrame(data =full_data_ss)

  return (simpa_dataset_ls, simpa_dataset_ss)

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