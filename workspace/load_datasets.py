import os
import pandas as pd
import glob
import numpy as np
import pickle
import urllib.request
import requests
import tarfile
import zipfile
import en_core_web_sm
from pyunpack import Archive


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
    labels = ['test'] * len(asset_test_orig) * len(asset_test_simps) + ['valid'] * len(asset_valid_orig) * len(asset_test_simps)

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

    full_data = {'ds_id' : 'ASSET', 'src' : src, 'src_id' : src_ids, 'simp' : simp, 'simp_id' : simp_ids, 'label' : labels, 'origin' : origin, 'granularity': 'sentence'}

    asset_dataset = pd.DataFrame(data=full_data)

    with open(asset_path + '/asset.pkl', 'wb') as f:
      pickle.dump(asset_dataset, f)

    #todo: metadata for dataset
  else:
    asset_dataset = pd.read_pickle(path_to_datasets + '/asset/asset.pkl')

  return asset_dataset

def load_htss_ds():
  # HTSS DATASET
  # https://github.com/slab-itu/HTSS/

  if not os.path.isfile(path_to_datasets + '/htss/htss.pkl'):
    htss_path = path_to_datasets + 'htss'
    htss_link = 'https://github.com/slab-itu/HTSS/'

    if not os.path.isdir(htss_path):
      os.mkdir(htss_path)
      os.chdir(htss_path)
      clone = 'git clone ' + htss_link
      os.system(clone)

    htss_file = htss_path + '/HTSS/data/new_data_set.csv'

    htss_simp_df = pd.read_csv(htss_file, encoding='latin1', header=0)
    htss_simp_df = htss_simp_df.reset_index()

    src_ids = []
    src_title = []
    src = []

    simp_ids = []
    simp_title = []
    simp = []

    for index, row in htss_simp_df.iterrows():
      source = htss_path + '/HTSS/data/' + row['Full_Paper_XML'][:-3] + 'txt'


      if os.path.exists(source):
        simp_title.append(row['Eureka_Title_Simplified'])
        simp.append(row['Eureka_Text_Simplified'])
        src_title.append(row['Paper_Title'])
        simp_ids.append(len(simp_ids))

        with open(source) as f:
          lines = f.readlines()
          # first four lines are ID and title of paper and can thus be ignored
          doc = str(lines[4:])[2:-2].replace('\\n', '').replace('", "', "', '").replace('", ', "', ").replace(', "', ", '").replace("', '", '').replace('\\r', '').replace('.  #@NEW_LINE#@#  ', '. ').replace('#@NEW_LINE#@#', '\\n').replace('  \\n  ', ' \\n ').replace('"', '').replace("'", '')
          src.append(doc)
          src_ids.append(len(src_ids))

    full_data = {'ds_id' : 'HTSS', 'src' : src, 'src_id' : src_ids, 'simp' : simp, 'simp_id' : simp_ids, 'src_title': src_title, 'simp_title': simp_title, 'granularity': 'document'}

    htss_dataset = pd.DataFrame(data=full_data)

    with open(htss_path + '/htss.pkl', 'wb') as f:
      pickle.dump(htss_dataset, f)
    
    #todo: metadata for dataset
  else:  
    htss_dataset = pd.read_pickle(path_to_datasets + 'htss/htss.pkl')

  return htss_dataset

def load_britannica_ds():
  # britannica dataset
  # http://www.cs.columbia.edu/~noemie/alignment/

  if not os.path.isfile(path_to_datasets + '/britannica/britannica.pkl'):
    britannica_path = path_to_datasets + 'britannica'
    britannica_links = ['http://www.cs.columbia.edu/~noemie/alignment/data/train/hum/baghdad-hum.txt', 
                        'http://www.cs.columbia.edu/~noemie/alignment/data/train/hum/bangkok-hum.txt', 
                        'http://www.cs.columbia.edu/~noemie/alignment/data/train/hum/budapest-hum.txt',
                        'http://www.cs.columbia.edu/~noemie/alignment/data/train/hum/jakart-hum.txt',
                        'http://www.cs.columbia.edu/~noemie/alignment/data/train/hum/kiev-hum.txt',
                        'http://www.cs.columbia.edu/~noemie/alignment/data/train/hum/lisbon-hum.txt',
                        'http://www.cs.columbia.edu/~noemie/alignment/data/train/hum/madrid-hum.txt',
                        'http://www.cs.columbia.edu/~noemie/alignment/data/train/hum/mexico-hum.txt',
                        'http://www.cs.columbia.edu/~noemie/alignment/data/train/hum/petersbourg-hum.txt',
                        'http://www.cs.columbia.edu/~noemie/alignment/data/train/hum/seoul-hum.txt',
                        'http://www.cs.columbia.edu/~noemie/alignment/data/train/hum/warsaw-hum.txt',
                        'http://www.cs.columbia.edu/~noemie/alignment/data/test/hum/buenos-hum.txt',
                        'http://www.cs.columbia.edu/~noemie/alignment/data/test/hum/caracas-hum.txt',
                        'http://www.cs.columbia.edu/~noemie/alignment/data/test/hum/damascus-hum.txt',
                        'http://www.cs.columbia.edu/~noemie/alignment/data/test/hum/dublin-hum.txt',
                        'http://www.cs.columbia.edu/~noemie/alignment/data/test/hum/havana-hum.txt',
                        'http://www.cs.columbia.edu/~noemie/alignment/data/test/hum/lima-hum.txt',
                        'http://www.cs.columbia.edu/~noemie/alignment/data/test/hum/manila-hum.txt',
                        'http://www.cs.columbia.edu/~noemie/alignment/data/test/hum/prague-hum.txt',
                        'http://www.cs.columbia.edu/~noemie/alignment/data/test/hum/vienna-hum.txt']

    if not os.path.isdir(britannica_path):
      os.mkdir(britannica_path)
      os.mkdir(britannica_path + '/train')
      os.mkdir(britannica_path + '/test')
      
      for link in britannica_links:
        if link[50:54] == 'test':
          f =  open(britannica_path + '/test/' + link[59:], 'w')
          for line in urllib.request.urlopen(link):
            f.write(line.decode('utf-8'))
        else:
          f = open(britannica_path + '/train/' + link[60:], 'w')
          for line in urllib.request.urlopen(link):
            f.write(line.decode('utf-8'))

    src_ids = []
    src = []
    simp = []
    simp_ids = []
    topics = []
    labels = []

    britannica_files_train = sorted(glob.glob(f"{britannica_path}/train/*"))
    britannica_files_test = sorted(glob.glob(f"{britannica_path}/test/*"))

    for path in britannica_files_train:
      with open(path) as f:
        lines = f.readlines()
        for l_id in range(0, len(lines), 3):
          if len(lines[l_id].strip()) > 0 and len(lines[l_id + 1].strip()) > 0:
            # filter out starting digits
            dig_id_source = 0
            while lines[l_id][dig_id_source].isdigit():
              dig_id_source = dig_id_source + 1
            dig_id_simp = 0
            while lines[l_id + 1][dig_id_simp].isdigit():
              dig_id_simp = dig_id_simp + 1

            if dig_id_source > 0:
              dig_id_source += 1
              dig_id_simp += 1  
  
            src.append(lines[l_id][dig_id_source:].replace(' .', '.').replace(' ,', ',').replace('( ', '(').replace(' )', ')'))
            simp.append(lines[l_id + 1][dig_id_simp:].replace(' .', '.').replace(' ,', ',').replace('( ', '(').replace(' )', ')'))

            labels.append('train')
            topics.append(path[len(britannica_path) + 7:-8])
            src_ids.append(len(src_ids))
            simp_ids.append(len(simp_ids))
    
    for path in britannica_files_test:
      with open(path) as f:
        lines = f.readlines()
        for l_id in range(0, len(lines), 3):
          if len(lines[l_id].strip()) > 0 and len(lines[l_id + 1].strip()) > 0:
            # filter out starting digits
            dig_id_source = 0
            while lines[l_id][dig_id_source].isdigit():
              dig_id_source = dig_id_source + 1
            dig_id_simp = 0
            while lines[l_id + 1][dig_id_simp].isdigit():
              dig_id_simp = dig_id_simp + 1

            if dig_id_source > 0:
              dig_id_source += 1
              dig_id_simp += 1  
  
            src.append(lines[l_id][dig_id_source:].replace(' .', '.').replace(' ,', ',').replace('( ', '(').replace(' )', ')').replace('\n', ''))
            simp.append(lines[l_id + 1][dig_id_simp:].replace(' .', '.').replace(' ,', ',').replace('( ', '(').replace(' )', ')').replace('\n', ''))

            labels.append('test')
            topics.append(path[len(britannica_path) + 6:-8])
            src_ids.append(len(src_ids))
            simp_ids.append(len(simp_ids))  

    full_data = {'ds_id' : 'britannica', 'src' : src, 'src_id' : src_ids, 'simp' : simp, 'simp_id' : simp_ids, 'label' : labels, 'topic': topics, 'granularity': 'sentence'}

    britannica_dataset = pd.DataFrame(data = full_data)

    with open(britannica_path + '/britannica.pkl', 'wb') as f:
      pickle.dump(britannica_dataset, f)
    
    #todo: metadata for dataset
  else:  
    britannica_dataset = pd.read_pickle(path_to_datasets + 'britannica/britannica.pkl')

  return britannica_dataset

def load_simpa_ds():
  # SIMPA DATASET
  # https://github.com/simpaticoproject/simpa

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

    full_data = {'ds_id' : 'simpa', 'src' : src, 'src_id' : src_ids, 'simp' : simp, 'simp_id' : simp_ids, 'origin' : origin, 'granularity': 'sentence'}

    simpa_dataset = pd.DataFrame(data = full_data)

    with open(simpa_path + '/simpa.pkl', 'wb') as f:
      pickle.dump(simpa_dataset, f)

    #todo: metadata for dataset
  else:
    simpa_dataset = pd.read_pickle(path_to_datasets + '/simpa/simpa.pkl')

  return simpa_dataset

def load_pwkp_ds():
  # PWKP dataset
  # https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2447

  if not os.path.isfile(path_to_datasets + '/pwkp/pwkp.pkl'): 
    pwkp_path = path_to_datasets + 'pwkp'
    pwkp_link = 'https://tudatalib.ulb.tu-darmstadt.de/bitstream/handle/tudatalib/2447/PWKP_108016.tar.gz?sequence=1&isAllowed=y'

    if not os.path.isdir(pwkp_path):
      os.mkdir(pwkp_path)
      
    response = requests.get(pwkp_link, stream=True)
    
    if response.status_code == 200:
      with open(pwkp_path + '/data.tar.gz', 'wb') as f:
          f.write(response.raw.read())

      f = tarfile.open(pwkp_path + '/data.tar.gz')
      f.extractall(pwkp_path)
      f.close()

    src_ids = []
    src = []
    simp_ids = []
    simp = []
    curr_src_id = -1

    with open(pwkp_path + '/PWKP_108016') as f:
      lines = f.readlines()
      # first: complex sentence, then up to multiple simplified ones
      source_sent = ''
      simp_sents = ''

      last_line_blank = True
      for l in lines:
        if len(l.strip()) > 0:
          if last_line_blank:
            source_sent = l.strip()
            curr_src_id = curr_src_id + 1
            last_line_blank = False
          else:
            simp_sents = simp_sents + l.strip() + ' '

        else:
          if len(simp_sents) > 0 and len(source_sent) > 0:
            simp.append(simp_sents)
            src.append(source_sent)

            src_ids.append(curr_src_id)
            simp_ids.append(len(simp_ids))

          last_line_blank = True
          simp_sents = ''
          source_sent = ''

    full_data = {'ds_id' : 'PWKP', 'src' : src, 'src_id' : src_ids, 'simp' : simp, 'simp_id' : simp_ids, 'granularity': 'sentence'}
    pwkp_dataset = pd.DataFrame(data = full_data)

    with open(pwkp_path + '/pwkp.pkl', 'wb') as f:
      pickle.dump(pwkp_dataset, f)

    #todo: metadata for dataset
  else:
    pwkp_dataset = pd.read_pickle(path_to_datasets + '/pwkp/pwkp.pkl')
  
  return pwkp_dataset

def load_benchls_ds():
  # BenchLS dataset
  # ghpaetzold.github.io/data/BenchLS.zip

  if not os.path.isfile(path_to_datasets + '/benchls/benchls.pkl'): 
    benchls_path = path_to_datasets + 'benchls'
    benchls_link = 'https://ghpaetzold.github.io/data/BenchLS.zip'
   
    if not os.path.isdir(benchls_path):
      os.mkdir(benchls_path)
      
    response = requests.get(benchls_link, stream=True)
    
    if response.status_code == 200:
      with open(benchls_path + '/data.zip', 'wb') as f:
          f.write(response.raw.read())

      with zipfile.ZipFile(benchls_path + '/data.zip', 'r') as f:
        f.extractall(benchls_path)

    src_ids = []
    src = []
    simp_ids = []
    simp = []

    nlp = en_core_web_sm.load()

    curr_src_id = -1

    with open(benchls_path + '/BenchLS.txt') as f:
      lines = f.readlines()
      # complex sentence, TAB, word to substitute, TAB, multiple simplified words
      for l in lines:

        pts = l.split('\t')
        replace_word = pts[1]
        replace_pos = int(pts[2])

        doc = nlp(pts[0])
        token = [token.text for token in doc]
        simps = []
        for i in range(3, len(pts)):
          token[replace_pos] = pts[i].replace('\n', '').split(':')[1]
          simps.append(' '.join(token).replace(' ,', ',').replace(' .', '.').replace(' :', ':').replace(' ;', ';').replace(' ?', '?').replace(' !', '!'))

        for i in range(len(simps)):
          src.append(pts[0])
          src_ids.append(curr_src_id + 1)
          simp.append(simps[i])
          simp_ids.append(len(simp_ids))

        curr_src_id = curr_src_id + 1

      full_data = {'ds_id' : 'BenchLS', 'src' : src, 'src_id' : src_ids, 'simp' : simp, 'simp_id' : simp_ids, 'granularity': 'sentence'}
      benchls_dataset = pd.DataFrame(data = full_data)

      with open(benchls_path + '/benchls.pkl', 'wb') as f:
        pickle.dump(benchls_dataset, f)

      #todo: metadata for dataset
  else:
    benchls_dataset = pd.read_pickle(path_to_datasets + '/benchls/benchls.pkl')
  return benchls_dataset

def load_dwikipedia_ds():
  # D-Wikipedia dataset
  # https://github.com/RLSNLP/Document-level-text-simplification

  if not os.path.isfile(path_to_datasets + '/dwikipedia/dwikipedia.pkl'):
    dwikipedia_path = path_to_datasets + 'dwikipedia'
    dwikipedia_link = 'https://github.com/RLSNLP/Document-level-text-simplification'

    if not os.path.isdir(dwikipedia_path):
      os.mkdir(dwikipedia_path)
      os.chdir(dwikipedia_path)
      clone = 'git clone ' + dwikipedia_link
      os.system(clone)

      Archive(dwikipedia_path + '/Document-level-text-simplification/Dataset/train.src.7z').extractall(dwikipedia_path + '/Document-level-text-simplification/Dataset')
      Archive(dwikipedia_path + '/Document-level-text-simplification/Dataset/train.tgt.7z').extractall(dwikipedia_path + '/Document-level-text-simplification/Dataset')
      
      os.remove(dwikipedia_path + '/Document-level-text-simplification/Dataset/train.src.7z')
      os.remove(dwikipedia_path + '/Document-level-text-simplification/Dataset/train.tgt.7z')

    dwikipedia_files = sorted(glob.glob(f"{dwikipedia_path}/Document-level-text-simplification/Dataset/*"))

    src_ids = []
    src = []
    simp_ids = []
    simp = []
    labels = []

    for f in dwikipedia_files:
      with open(f, encoding='latin1') as f:
        lines = f.readlines()
        label = f[len(dwikipedia_files)+1:-4]
        if f[-3:] == 'src':
          for l in lines:
            src.append(l)
            labels.append(label)
            src_ids.append(len(src_ids))
        else:
          for l in lines:
            simp.append(l)
            simp_ids.append(len(simp_ids))

    full_data = {'ds_id' : 'D-Wikipedia', 'src' : src, 'src_id' : src_ids, 'simp' : simp, 'simp_id' : simp_ids, 'label': labels, 'granularity': 'sentence'}
    dwikipedia_dataset = pd.DataFrame(data = full_data)

    with open(dwikipedia_path + '/dwikipedia.pkl', 'wb') as f:
      pickle.dump(dwikipedia_dataset, f)

    #todo: metadata for dataset
  else:
    dwikipedia_dataset = pd.read_pickle(path_to_datasets + '/dwikipedia/dwikipedia.pkl')
  return dwikipedia_dataset

def load_massalign_ds():
  # massalign dataset
  # https://github.com/stefanpaun/massalign

  if not os.path.isfile(path_to_datasets + '/massalign/massalign.pkl'):
    massalign_path = path_to_datasets + 'massalign'
    massalign_link = 'https://github.com/stefanpaun/massalign'

    if not os.path.isdir(massalign_path):
      os.mkdir(massalign_path)
      os.chdir(massalign_path)
      clone = 'git clone ' + massalign_link
      os.system(clone)

    paragraph = massalign_path + '/massalign/dataset/dataset_paragraphs.txt'
    sentence = massalign_path + '/massalign/dataset/dataset_sentences.txt'

    src_ids = []
    src = []
    simp_ids = []
    simp = []
    granularity = []

    with open(paragraph) as f:
      lines = f.readlines()
      for l_id in range(0, len(lines), 3):
        if len(lines[l_id].strip()) > 0 and len(lines[l_id + 1].strip()) > 0:
          src.append(lines[l_id])
          simp.append(lines[l_id + 1])
          src_ids.append(len(src_ids))
          simp_ids.append(len(simp_ids))
          granularity.append('paragraph')

    with open(sentence) as f:
      lines = f.readlines()
      for l_id in range(0, len(lines), 3):
        if len(lines[l_id].strip()) > 0 and len(lines[l_id + 1].strip()) > 0:
          src.append(lines[l_id])
          simp.append(lines[l_id + 1])
          src_ids.append(len(src_ids))
          simp_ids.append(len(simp_ids))
          granularity.append('sentence')

    full_data = {'ds_id' : 'massalign', 'src' : src, 'src_id' : src_ids, 'simp' : simp, 'simp_id' : simp_ids, 'granularity': granularity}
    massalign_dataset = pd.DataFrame(data = full_data)

    with open(massalign_path + '/massalign.pkl', 'wb') as f:
      pickle.dump(massalign_dataset, f)

    #todo: metadata for dataset
  else:
    massalign_dataset = pd.read_pickle(path_to_datasets + '/massalign/massalign.pkl')
  return massalign_dataset

def load_metaeval_ds():
  # metaeval DATASET
  # https://github.com/feralvam/metaeval-simplification

  if not os.path.isfile(path_to_datasets + '/metaeval/metaeval.pkl'):
    metaeval_path = path_to_datasets + 'metaeval'
    metaeval_link = 'https://github.com/feralvam/metaeval-simplification'

    if not os.path.isdir(metaeval_path):
      os.mkdir(metaeval_path)
      os.chdir(metaeval_path)
      clone = 'git clone ' + metaeval_link
      os.system(clone)

    metaeval_file = metaeval_path + '/metaeval-simplification/data/simplicity_DA.csv'

    metaeval_df = pd.read_csv(metaeval_file, encoding='latin1', header=0)
    metaeval_df = metaeval_df.reset_index()

    src_ids = []
    src = []
    simp_ids = []
    simp = []
    origin = []
    sent_ids = {}

    for index, row in metaeval_df.iterrows():
      if str(row['sent_id']) in sent_ids:
        curr_src_id = sent_ids[str(row['sent_id'])]
        src_ids.append(curr_src_id)
        src.append(row['orig_sent'])
        simp_ids.append(len(simp_ids))
        simp.append(row['simp_sent'])
        origin.append(row['sys_name'])
      else:
        curr_src_id = len(sent_ids)
        sent_ids[str(row['sent_id'])] = curr_src_id
        src_ids.append(curr_src_id)
        src.append(row['orig_sent'])
        simp_ids.append(len(simp_ids))
        simp.append(row['simp_sent'])
        origin.append(row['sys_name'])

    full_data = {'ds_id' : 'metaeval', 'src' : src, 'src_id' : src_ids, 'simp' : simp, 'simp_id' : simp_ids, 'origin': origin, 'granularity': 'sentence'}
    metaeval_dataset = pd.DataFrame(data = full_data)

    with open(metaeval_path + '/metaeval.pkl', 'wb') as f:
      pickle.dump(metaeval_dataset, f)

    #todo: metadata for dataset
  else:
    metaeval_dataset = pd.read_pickle(path_to_datasets + '/metaeval/metaeval.pkl')
  return metaeval_dataset

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
  ds = load_metaeval_ds()

if __name__ == '__main__':
  main()
