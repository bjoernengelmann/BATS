from math import floor
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
import gzip
from bs4 import BeautifulSoup
import py7zr

path_to_datasets = '/workspace/datasets/'

def load_list_file(path):
  data = []
  with open('/' + path, "r") as f:
    for line in f.readlines():
      data.append(line.strip())
  return data

def load_asset_ds():
  # ASSET DATASET
  # https://github.com/facebookresearch/asset

  if not os.path.isfile(path_to_datasets + 'asset/asset.pkl'):
    asset_path = path_to_datasets + 'asset'
    asset_link = 'https://github.com/facebookresearch/asset'

    print(os.path.dirname(__file__))

    if not os.path.isdir(asset_path):
      os.mkdir(asset_path)
      os.chdir(asset_path)
      clone = 'git clone ' + asset_link
      os.system(clone)

      asset_files = sorted(glob.glob(f"/{asset_path}/asset/dataset/*"))

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
      duplicated = []

      simp_ids = range(0, (len(asset_test_orig + asset_valid_orig)) * len(asset_test_simps)) 

      for i in range(len(asset_test_simps)):
          for j in asset_test_simps[i]:
            simp.append(j)
            origin.append('annotator_' + str(i))

      for i in range(len(asset_valid_simps)):
          for j in asset_valid_simps[i]:
            simp.append(j)
            origin.append('annotator_' + str(i))

      for i in range(len(simp)):
        if simp[i] == src[i]:
          duplicated.append(True)
        else: 
          duplicated.append(False)

      full_data = {'ds_id': 'ASSET', 'src': src, 'src_id': src_ids, 'simp': simp, 'simp_id': simp_ids, 'label': labels, 'origin': origin, 'granularity': 'sentence', 'duplicated': duplicated}
      asset_dataset = pd.DataFrame(data=full_data)

      with open('/' + asset_path + '/asset.pkl', 'wb') as f:
        pickle.dump(asset_dataset, f)
  else:
    asset_dataset = pd.read_pickle(path_to_datasets + 'asset/asset.pkl')

  return asset_dataset

def load_automets_ds():
  # AutoMeTS dataset
  # https://github.com/vanh17/MedTextSimplifier

  if not os.path.isfile(path_to_datasets + 'automets/automets.pkl'):
    automets_path = path_to_datasets + 'automets'
    automets_link = 'https://github.com/vanh17/MedTextSimplifier'

    if not os.path.isdir(automets_path):
      os.mkdir(automets_path)
      os.chdir(automets_path)
      clone = 'git clone ' + automets_link
      os.system(clone)

      src_ids = []
      src = []
      simp_ids = []
      simp = []
      duplicated = []

      # complex word, TAB, number, TAB, sentence
      with open('/' + automets_path + '/MedTextSimplifier/data_processing/data/NormalMedicalCorpora') as f:
        lines = f.readlines()
        for l in lines:
          pts = l.split('\t')
          src.append(pts[2].replace(' ,', ',').replace(' .', '.').replace(' :', ':').replace(' ;', ';').replace(' ?', '?').replace(' !', '!'))
          src_ids.append(len(src_ids))

      # complex word, TAB, number, TAB, sentence
      with open('/' + automets_path + '/MedTextSimplifier/data_processing/data/SimpleMedicalCorpora') as f:
        lines = f.readlines()
        for l in lines:
          pts = l.split('\t')
          simp.append(pts[2].replace(' ,', ',').replace(' .', '.').replace(' :', ':').replace(' ;', ';').replace(' ?', '?').replace(' !', '!'))
          simp_ids.append(len(simp_ids))

      # sentence
      with open('/' + automets_path + '/MedTextSimplifier/data_processing/data/normal.txt') as f:
        lines = f.readlines()
        for l in lines:
          src.append(l.replace(' -RRB- ', ' ').replace(' -LRB- ', ' ').replace(' -RSB- ', ' ').replace(' ,', ',').replace(' .', '.').replace(' :', ':').replace(' ;', ';').replace(' ?', '?').replace(' !', '!'))
          src_ids.append(len(src_ids))

      # sentence
      with open('/' + automets_path + '/MedTextSimplifier/data_processing/data/simple.txt') as f:
        lines = f.readlines()
        for l in lines:
          simp.append(l.replace(' -RRB- ', ' ').replace(' -LRB- ', ' ').replace(' -RSB- ', ' ').replace(' ,', ',').replace(' .', '.').replace(' :', ':').replace(' ;', ';').replace(' ?', '?').replace(' !', '!'))
          simp_ids.append(len(simp_ids))

      for i in range(len(simp)):
        if simp[i] == src[i]:
          duplicated.append(True)
        else: 
          duplicated.append(False)

      full_data = {'ds_id': 'AutoMeTS', 'src': src, 'src_id': src_ids, 'simp': simp, 'simp_id': simp_ids, 'granularity': 'sentence', 'duplicated': duplicated}
      automets_dataset = pd.DataFrame(data = full_data)

      with open('/' + automets_path + '/automets.pkl', 'wb') as f:
        pickle.dump(automets_dataset, f)
  else:
    automets_dataset = pd.read_pickle(path_to_datasets + 'automets/automets.pkl')
  return automets_dataset
        
def load_benchls_ds():
  # BenchLS dataset
  # ghpaetzold.github.io/data/BenchLS.zip

  if not os.path.isfile(path_to_datasets + 'benchls/benchls.pkl'): 
    benchls_path = path_to_datasets + 'benchls'
    benchls_link = 'https://ghpaetzold.github.io/data/BenchLS.zip'
   
    if not os.path.isdir(benchls_path):
      os.mkdir(benchls_path)
      
      response = requests.get(benchls_link, stream=True)
      
      if response.status_code == 200:
        with open('/' + benchls_path + '/data.zip', 'wb') as f:
            f.write(response.raw.read())

        with zipfile.ZipFile(benchls_path + '/data.zip', 'r') as f:
          f.extractall(benchls_path)

        src_ids = []
        src = []
        simp_ids = []
        simp = []
        duplicated = []

        nlp = en_core_web_sm.load()

        curr_src_id = -1

        with open('/' + benchls_path + '/BenchLS.txt') as f:
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
              if simps[i] != pts[0]:
                duplicated.append(False)
              else:
                duplicated.append(True)

              src.append(pts[0])
              src_ids.append(curr_src_id + 1)
              simp.append(simps[i])
              simp_ids.append(len(simp_ids))

            curr_src_id = curr_src_id + 1

          full_data = {'ds_id': 'BenchLS', 'src': src, 'src_id': src_ids, 'simp': simp, 'simp_id': simp_ids, 'granularity': 'sentence', 'duplicated': duplicated}
          benchls_dataset = pd.DataFrame(data = full_data)

          with open('/' + benchls_path + '/benchls.pkl', 'wb') as f:
            pickle.dump(benchls_dataset, f)
  else:
    benchls_dataset = pd.read_pickle(path_to_datasets + 'benchls/benchls.pkl')
  return benchls_dataset

def load_britannica_ds():
  # britannica dataset
  # http://www.cs.columbia.edu/~noemie/alignment/

  if not os.path.isfile(path_to_datasets + 'britannica/britannica.pkl'):
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
      duplicated = []

      britannica_files_train = sorted(glob.glob(f"{britannica_path}/train/*"))
      britannica_files_test = sorted(glob.glob(f"{britannica_path}/test/*"))

      for path in britannica_files_train:
        with open('/' + path) as f:
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
    
              src_hlp = lines[l_id][dig_id_source:].replace(' .', '.').replace(' ,', ',').replace('( ', '(').replace(' )', ')')
              simp_hlp = lines[l_id + 1][dig_id_simp:].replace(' .', '.').replace(' ,', ',').replace('( ', '(').replace(' )', ')')
              
              if src_hlp != simp_hlp:
                duplicated.append(False)
              else:
                duplicated.append(True)

              src.append(src_hlp)
              simp.append(simp_hlp)

              labels.append('train')
              topics.append(path[len(britannica_path) + 7:-8])
              src_ids.append(len(src_ids))
              simp_ids.append(len(simp_ids))
      
      for path in britannica_files_test:
        with open('/' + path) as f:
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
    
              src_hlp = lines[l_id][dig_id_source:].replace(' .', '.').replace(' ,', ',').replace('( ', '(').replace(' )', ')').replace('\n', '')
              simp_hlp = lines[l_id + 1][dig_id_simp:].replace(' .', '.').replace(' ,', ',').replace('( ', '(').replace(' )', ')').replace('\n', '')

              if src_hlp != simp_hlp:
                duplicated.append(False)
              else: 
                duplicated.append(True)

              src.append(src_hlp)
              simp.append(simp_hlp)

              labels.append('test')
              topics.append(path[len(britannica_path) + 6:-8])
              src_ids.append(len(src_ids))
              simp_ids.append(len(simp_ids))  

      full_data = {'ds_id': 'britannica', 'src': src, 'src_id': src_ids, 'simp': simp, 'simp_id': simp_ids, 'label': labels, 'topic': topics, 'granularity': 'sentence', 'duplicated': duplicated}
      britannica_dataset = pd.DataFrame(data = full_data)

      with open('/' + britannica_path + '/britannica.pkl', 'wb') as f:
        pickle.dump(britannica_dataset, f)
  else:  
    britannica_dataset = pd.read_pickle(path_to_datasets + 'britannica/britannica.pkl')

  return britannica_dataset

def load_dwikipedia_ds():
  # D-Wikipedia dataset
  # https://github.com/RLSNLP/Document-level-text-simplification

  if not os.path.isfile(path_to_datasets + 'dwikipedia/dwikipedia.pkl'):
    dwikipedia_path = path_to_datasets + 'dwikipedia'
    dwikipedia_link = 'https://github.com/RLSNLP/Document-level-text-simplification'

    if not os.path.isdir(dwikipedia_path):
      os.mkdir(dwikipedia_path)
      os.chdir(dwikipedia_path)
      clone = 'git clone ' + dwikipedia_link
      os.system(clone)

      archive = py7zr.SevenZipFile(dwikipedia_path + '/Document-level-text-simplification/Dataset/train.src.7z', mode='r')
      archive.extractall(path=dwikipedia_path + '/Document-level-text-simplification/Dataset')
      archive.close()
      
      archive = py7zr.SevenZipFile(dwikipedia_path + '/Document-level-text-simplification/Dataset/train.tgt.7z', mode='r')
      archive.extractall(path=dwikipedia_path + '/Document-level-text-simplification/Dataset')
      archive.close()
      
      os.remove(dwikipedia_path + '/Document-level-text-simplification/Dataset/train.src.7z')
      os.remove(dwikipedia_path + '/Document-level-text-simplification/Dataset/train.tgt.7z')

      dwikipedia_files = sorted(glob.glob(f"{dwikipedia_path}/Document-level-text-simplification/Dataset/*"))

      src_ids = []
      src = []
      simp_ids = []
      simp = []
      labels = []
      duplicated = []

      for f in dwikipedia_files:
        with open('/' + f, encoding='latin1') as f:
          lines = f.readlines()
          label = f.name[len(dwikipedia_files)+1:-4]
          if f.name[-3:] == 'src':
            for l in lines:
              src.append(l)
              labels.append(label)
              src_ids.append(len(src_ids))
          else:
            for l in lines:
              simp.append(l)
              simp_ids.append(len(simp_ids))

      for i in range(len(simp)):
        if simp[i] == src[i]:
          duplicated.append(True)
        else: 
          duplicated.append(False)

      full_data = {'ds_id': 'D-Wikipedia', 'src': src, 'src_id': src_ids, 'simp': simp, 'simp_id': simp_ids, 'label': labels, 'granularity': 'sentence', 'duplicated': duplicated}
      dwikipedia_dataset = pd.DataFrame(data = full_data)

      with open('/' + dwikipedia_path + '/dwikipedia.pkl', 'wb') as f:
        pickle.dump(dwikipedia_dataset, f)
  else:
    dwikipedia_dataset = pd.read_pickle(path_to_datasets + 'dwikipedia/dwikipedia.pkl')
  return dwikipedia_dataset

def load_ewsewgmpm_ds():
  # EW-SEW-gmpm dataset
  # https://github.com/senisioi/NeuralTextSimplification

  if not os.path.isfile(path_to_datasets + 'ewsewgmpm/ewsewgmpm.pkl'):
    ewsewgmpm_path = path_to_datasets + 'ewsewgmpm'
    ewsewgmpm_link = 'https://github.com/senisioi/NeuralTextSimplification'

    if not os.path.isdir(ewsewgmpm_path):
      os.mkdir(ewsewgmpm_path)
      os.chdir(ewsewgmpm_path)
      clone = 'git clone ' + ewsewgmpm_link
      os.system(clone)

      src_ids = []
      src = []
      simp_ids = []
      simp = []
      labels = []
      duplicated = []

      all_simp = []
      all_src = []
      all_labels = []

      ewsewgmpm_files = sorted(glob.glob(f"{ewsewgmpm_path}/NeuralTextSimplification/data/*"))

      for ewsewgmpm_file in ewsewgmpm_files:
        if os.path.isfile(ewsewgmpm_file):
          with open('/' + ewsewgmpm_file) as f:
            lines = f.readlines()

            for l in lines:
              if ewsewgmpm_file[-3:] == '.en':
                all_src.append(l.replace(' ,', ',').replace(' .', '.').replace(' :', ':').replace(' ;', ';').replace(' ?', '?').replace(' !', '!').replace('( ', '(').replace(' )', ')'))
              else:
                all_simp.append(l.replace(' ,', ',').replace(' .', '.').replace(' :', ':').replace(' ;', ';').replace(' ?', '?').replace(' !', '!').replace('( ', '(').replace(' )', ')'))

                if ewsewgmpm_file[len('ewsewgmpm_path}/NeuralTextSimplification/data/'):2] == 'dev':
                  all_labels.append('dev')
                elif ewsewgmpm_file[len('ewsewgmpm_path}/NeuralTextSimplification/data/'):2] == 'tes':
                  all_labels.append('test')
                else: 
                  all_labels.append('train')

      all_src_ids = {}        

      for i in range(len(all_simp)):
        
        if all_src[i] not in all_src_ids:
          all_src_ids[all_src[i]] = len(all_src_ids)
        src_ids.append(all_src_ids[all_src[i]])
        src.append(all_src[i])
        simp_ids.append(len(simp_ids))
        simp.append(all_simp[i])
        labels.append(all_labels[i])

        if all_simp[i] != all_src[i]:
          duplicated.append(False)
        else:
          duplicated.append(True)


      full_data = {'ds_id': 'EW-SEW-gmpm', 'src': src, 'src_id': src_ids, 'simp': simp, 'simp_id': simp_ids, 'label': labels, 'granularity': 'sentence', 'duplicated': duplicated}
      ewsewgmpm_dataset = pd.DataFrame(data=full_data)

      with open('/' + ewsewgmpm_path + '/ewsewgmpm.pkl', 'wb') as f:
        pickle.dump(ewsewgmpm_dataset, f)
  else:
    ewsewgmpm_dataset = pd.read_pickle(path_to_datasets + 'ewsewgmpm/ewsewgmpm.pkl')

  return ewsewgmpm_dataset

def load_ewsewturk_ds():
  # EW-SEW-MTurk
  # https://cs.pomona.edu/~dkauchak/simplification/lex.mturk.14/lex.mturk.14.tar.gz

  if not os.path.isfile(path_to_datasets + 'ewsewturk/ewsewturk.pkl'): 
    ewsewturk_path = path_to_datasets + 'ewsewturk'
    ewsewturk_link = 'https://cs.pomona.edu/~dkauchak/simplification/lex.mturk.14/lex.mturk.14.tar.gz'

    if not os.path.isdir(ewsewturk_path):
      os.mkdir(ewsewturk_path)
      
    response = requests.get(ewsewturk_link, stream=True)
    
    if response.status_code == 200:
      with open('/' + ewsewturk_path + '/data.tar.gz', 'wb') as f:
          f.write(response.raw.read())

      f = tarfile.open(ewsewturk_path + '/data.tar.gz')
      f.extractall(ewsewturk_path)
      f.close()

      src_ids = []
      src = []
      simp_ids = []
      simp = []
      duplicated = []

      nlp = en_core_web_sm.load()

      curr_src_id = -1

      with open('/' + ewsewturk_path + '/lex.mturk.14/lex.mturk.txt', encoding='latin1') as f:
        lines = f.readlines()[1:]
        # complex sentence, TAB, word to substitute, TAB, multiple simplified words
        for l in lines:
          pts = l.split('\t')
          replace_word = pts[1]
          replace_pos = -1
          doc = nlp(pts[0])
          token = [token.text for token in doc]
          simps = []
          curr_simps = []

          for i in range(len(token)):
            if token[i] == replace_word:
              replace_pos = i
              break

          for i in range(2, len(pts)):
            if pts[i] not in curr_simps:
              token[replace_pos] = pts[i]
              simps.append(' '.join(token).replace(' ,', ',').replace(' .', '.').replace(' :', ':').replace(' ;', ';').replace(' ?', '?').replace(' !', '!'))
              curr_simps.append(pts[i])

          for i in range(len(simps)):
            src.append(pts[0].replace(' ,', ',').replace(' .', '.').replace(' :', ':').replace(' ;', ';').replace(' ?', '?').replace(' !', '!'))
            src_ids.append(curr_src_id + 1)
            simp.append(simps[i])
            simp_ids.append(len(simp_ids))

          curr_src_id = curr_src_id + 1

      for i in range(len(simp)):
        if simp[i] == src[i]:
          duplicated.append(True)
        else: 
          duplicated.append(False)

      full_data = {'ds_id': 'EW-SEW-Turk', 'src': src, 'src_id': src_ids, 'simp': simp, 'simp_id': simp_ids, 'origin': 'Turker', 'granularity': 'sentence', 'duplicated': duplicated}
      ewsewturk_dataset = pd.DataFrame(data=full_data)

      with open('/' + ewsewturk_path + '/ewsewturk.pkl', 'wb') as f:
        pickle.dump(ewsewturk_dataset, f)
  else:
    ewsewturk_dataset = pd.read_pickle(path_to_datasets + 'ewsewturk/ewsewturk.pkl')

  return ewsewturk_dataset

def load_htss_ds():
  # HTSS DATASET
  # https://github.com/slab-itu/HTSS/

  if not os.path.isfile(path_to_datasets + 'htss/htss.pkl'):
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
      duplicated = []

      for index, row in htss_simp_df.iterrows():
        source = htss_path + '/HTSS/data/' + row['Full_Paper_XML'][:-3] + 'txt'

        if os.path.exists(source):
          simp_hlp = row['Eureka_Text_Simplified']

          with open('/' + source) as f:
            lines = f.readlines()
            # first four lines are ID and title of paper and can thus be ignored
            doc = str(lines[4:])[2:-2].replace('\\n', '').replace('", "', "', '").replace('", ', "', ").replace(', "', ", '").replace("', '", '').replace('\\r', '').replace('.  #@NEW_LINE#@#  ', '. ').replace('#@NEW_LINE#@#', '\\n').replace('  \\n  ', ' \\n ').replace('"', '').replace("'", '')
            
            if doc != simp_hlp:
              duplicated.append(False)
            else:
              duplicated.append(True)

            src.append(doc)
            src_ids.append(len(src_ids))

            simp.append(simp_hlp)
            simp_title.append(row['Eureka_Title_Simplified'])
            src_title.append(row['Paper_Title'])
            simp_ids.append(len(simp_ids))

      full_data = {'ds_id': 'HTSS', 'src': src, 'src_id': src_ids, 'simp': simp, 'simp_id': simp_ids, 'src_title': src_title, 'simp_title': simp_title, 'granularity': 'document', 'duplicated': duplicated}

      htss_dataset = pd.DataFrame(data=full_data)

      with open('/' + htss_path + '/htss.pkl', 'wb') as f:
        pickle.dump(htss_dataset, f)
  else:  
    htss_dataset = pd.read_pickle(path_to_datasets + 'htss/htss.pkl')

  return htss_dataset

def load_hutssf_ds():
  # HutSSF dataset
  # https://cs.pomona.edu/~dkauchak/simplification/human_simplification.data.zip

  if not os.path.isfile(path_to_datasets + 'hutssf/hutssf.pkl'):
    hutssf_path = path_to_datasets + 'hutssf'
    hutssf_link = 'https://cs.pomona.edu/~dkauchak/simplification/human_simplification.data.zip'

    if not os.path.isdir(hutssf_path):
      os.mkdir(hutssf_path)
      
    response = requests.get(hutssf_link, stream=True)
    
    if response.status_code == 200:
      with open('/' + hutssf_path + '/data.zip', 'wb') as f:
          f.write(response.raw.read())

      with zipfile.ZipFile(hutssf_path + '/data.zip', 'r') as f:
        f.extractall(hutssf_path)

      hutssf_files = sorted(glob.glob(f"{hutssf_path}/data/*"))

      src_ids = []
      src = []
      simp_ids = []
      simp = []
      origins = []
      labels = []
      duplicated = []
      sent_ids = {}

      for hutssf_file in hutssf_files:
        if hutssf_file[-3:] == 'csv':
          hutssf_df = pd.read_csv(hutssf_file, encoding='latin1', header=0)
        
          for index, row in hutssf_df.iterrows():
            if str(row['Original']) in sent_ids:
              curr_src_id = sent_ids[str(row['Original'])]
              src_ids.append(curr_src_id)
              src.append(row['Original'])
              simp_ids.append(len(simp_ids))
              simp.append(row['Simplified'])
              origins.append(row['Source'])
              labels.append(hutssf_file[len(hutssf_path + '/data/'):-4])
            else:
              curr_src_id = len(sent_ids)
              sent_ids[str(row['Original'])] = curr_src_id
              src_ids.append(curr_src_id)
              src.append(row['Original'])
              simp_ids.append(len(simp_ids))
              simp.append(row['Simplified'])
              origins.append(row['Source'])
              labels.append(hutssf_file[len(hutssf_path + '/data/'):-4])

            if row['Original'] != row['Simplified']:
              duplicated.append(False)
            else:
              duplicated.append(True)

      full_data = {'ds_id': 'HutSSF', 'src': src, 'src_id': src_ids, 'simp': simp, 'simp_id': simp_ids, 'label': labels, 'origin': origins, 'granularity': 'sentence', 'duplicated': duplicated}
      hutssf_dataset = pd.DataFrame(data = full_data)

      with open('/' + hutssf_path + '/hutssf.pkl', 'wb') as f:
        pickle.dump(hutssf_dataset, f)
  else:
    hutssf_dataset = pd.read_pickle(path_to_datasets + 'hutssf/hutssf.pkl')
  return hutssf_dataset

def load_massalign_ds():
  # massalign dataset
  # https://github.com/stefanpaun/massalign

  if not os.path.isfile(path_to_datasets + 'massalign/massalign.pkl'):
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
      duplicated = []

      with open('/' + paragraph) as f:
        lines = f.readlines()
        for l_id in range(0, len(lines), 3):
          if len(lines[l_id].strip()) > 0 and len(lines[l_id + 1].strip()) > 0:
            src_hlp = lines[l_id]
            simp_hlp = lines[l_id + 1]

            if src_hlp != simp_hlp:
              duplicated.append(False)
            else:
              duplicated.append(True)

            src.append(src_hlp)
            simp.append(simp_hlp)
            src_ids.append(len(src_ids))
            simp_ids.append(len(simp_ids))
            granularity.append('paragraph')

      with open('/' + sentence) as f:
        lines = f.readlines()
        for l_id in range(0, len(lines), 3):
          if len(lines[l_id].strip()) > 0 and len(lines[l_id + 1].strip()) > 0:
            src_hlp = lines[l_id]
            simp_hlp = lines[l_id + 1]

            if src_hlp != simp_hlp:
                duplicated.append(False)
            else:
              duplicated.append(True)

            src.append(src_hlp)
            simp.append(simp_hlp)
            src_ids.append(len(src_ids))
            simp_ids.append(len(simp_ids))
            granularity.append('sentence')

      full_data = {'ds_id': 'massalign', 'src': src, 'src_id': src_ids, 'simp': simp, 'simp_id': simp_ids, 'granularity': granularity, 'duplicated': duplicated}
      massalign_dataset = pd.DataFrame(data = full_data)

      with open('/' + massalign_path + '/massalign.pkl', 'wb') as f:
        pickle.dump(massalign_dataset, f)
  else:
    massalign_dataset = pd.read_pickle(path_to_datasets + 'massalign/massalign.pkl')
  return massalign_dataset

def load_metaeval_ds():
  # metaeval DATASET
  # https://github.com/feralvam/metaeval-simplification

  if not os.path.isfile(path_to_datasets + 'metaeval/metaeval.pkl'):
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
      duplicated = []
      sent_ids = {}

      for index, row in metaeval_df.iterrows():
        if str(row['sent_id']) in sent_ids:
          src_hlp = row['orig_sent']
          simp_hlp = row['simp_sent']

          if src_hlp != simp_hlp:
            duplicated.append(False)
          else:
            duplicated.append(True)

          curr_src_id = sent_ids[str(row['sent_id'])]
          src_ids.append(curr_src_id)
          src.append(src_hlp)
          simp_ids.append(len(simp_ids))
          simp.append(simp_hlp)
          origin.append(row['sys_name'])
        else:
          src_hlp = row['orig_sent']
          simp_hlp = row['simp_sent']
        
          if src_hlp != simp_hlp:
            duplicated.append(False)
          else:
            duplicated.append(True)
          curr_src_id = len(sent_ids)
          sent_ids[str(row['sent_id'])] = curr_src_id
          src_ids.append(curr_src_id)
          src.append(src_hlp)
          simp_ids.append(len(simp_ids))
          simp.append(simp_hlp)
          origin.append(row['sys_name'])

      full_data = {'ds_id': 'metaeval', 'src': src, 'src_id': src_ids, 'simp': simp, 'simp_id': simp_ids, 'origin': origin, 'granularity': 'sentence', 'duplicated': duplicated}
      metaeval_dataset = pd.DataFrame(data = full_data)

      with open('/' + metaeval_path + '/metaeval.pkl', 'wb') as f:
        pickle.dump(metaeval_dataset, f)
  else:
    metaeval_dataset = pd.read_pickle(path_to_datasets + 'metaeval/metaeval.pkl')
  return metaeval_dataset

def load_mturksf_ds():
  # MTurkSF dataset
  # https://cs.pomona.edu/~dkauchak/simplification/word_filtering/MTurk_annotations.zip

  if not os.path.isfile(path_to_datasets + 'mturksf/mturksf.pkl'): 
    mturksf_path = path_to_datasets + 'mturksf'
    mturksf_link = 'https://cs.pomona.edu/~dkauchak/simplification/word_filtering/MTurk_annotations.zip'
  
    if not os.path.isdir(mturksf_path):
      os.mkdir(mturksf_path)
      
    response = requests.get(mturksf_link, stream=True)
    
    if response.status_code == 200:
      with open('/' + mturksf_path + '/data.zip', 'wb') as f:
          f.write(response.raw.read())

      with zipfile.ZipFile(mturksf_path + '/data.zip', 'r') as f:
        f.extractall(mturksf_path)

      src_ids = []
      src = []
      simp_ids = []
      simp = []
      topics = []
      duplicated = []

      mturksf_df = pd.read_excel(mturksf_path + '/MTurk_annotations/MTurk-annotations.xlsx', sheet_name=1)

      ids = {}

      for index, row in mturksf_df.iterrows():
        if row['rating'] == 1:
          txt = row['sentence'].replace('<b>' + row['original_word'] + '</b>', row['original_word'])

          if txt not in ids:
            ids[txt] = len(ids)
          
          simp_hlp = row['sentence'].replace('<b>' + row['original_word'] + '</b>', row['replacement_word'])

          if txt != simp_hlp:
            duplicated.append(False)
          else:
            duplicated.append(True)

          src_ids.append(ids[txt])
          src.append(txt)
          simp_ids.append(len(simp_ids))
          simp.append(simp_hlp)
          topics.append(row['source'][:-4])

      
      full_data = {'ds_id': 'MTurkSF', 'src': src, 'src_id': src_ids, 'simp': simp, 'simp_id': simp_ids, 'topic': topics, 'granularity': 'sentence', 'duplicated': duplicated}
      mturksf_dataset = pd.DataFrame(data = full_data)

      with open('/' + mturksf_path + '/mturksf.pkl', 'wb') as f:
        pickle.dump(mturksf_dataset, f)
  else:
    mturksf_dataset = pd.read_pickle(path_to_datasets + 'mturksf/mturksf.pkl')
  return mturksf_dataset

def load_nnseval_ds():
  # NNSeval dataset
  # ghpaetzold.github.io/data/NNSeval.zip

  if not os.path.isfile(path_to_datasets + 'nnseval/nnseval.pkl'): 
    nnseval_path = path_to_datasets + 'nnseval'
    nnseval_link = 'https://ghpaetzold.github.io/data/NNSeval.zip'
  
    if not os.path.isdir(nnseval_path):
      os.mkdir(nnseval_path)
      
    response = requests.get(nnseval_link, stream=True)
    
    if response.status_code == 200:
      with open('/' + nnseval_path + '/data.zip', 'wb') as f:
          f.write(response.raw.read())

      with zipfile.ZipFile(nnseval_path + '/data.zip', 'r') as f:
        f.extractall(nnseval_path)

      src_ids = []
      src = []
      simp_ids = []
      simp = []
      duplicated = []
      
      nlp = en_core_web_sm.load()

      curr_src_id = -1

      with open('/' + nnseval_path + '/NNSeval.txt') as f:
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
            src_hlp = pts[0]
            simp_hlp = simps[i]

            if src_hlp != simps[i]:
              duplicated.append(False)
            else:
              duplicated.append(True)

            src.append(pts[0])
            src_ids.append(curr_src_id + 1)
            simp.append(simps[i])
            simp_ids.append(len(simp_ids))

          curr_src_id = curr_src_id + 1

      full_data = {'ds_id': 'NNSeval', 'src': src, 'src_id': src_ids, 'simp': simp, 'simp_id': simp_ids, 'granularity': 'sentence', 'duplicated': duplicated}
      nnseval_dataset = pd.DataFrame(data = full_data)

      with open('/' + nnseval_path + '/nnseval.pkl', 'wb') as f:
        pickle.dump(nnseval_dataset, f)
  else:
    nnseval_dataset = pd.read_pickle(path_to_datasets + 'nnseval/nnseval.pkl')
  return nnseval_dataset

def load_onestopenglish_ds():
  # OneStopEnglish dataset
  # https://github.com/nishkalavallabhi/OneStopEnglishCorpus/

  if not os.path.isfile(path_to_datasets + 'onestopenglish/onestopenglish.pkl'): 
    onestopenglish_path = path_to_datasets + 'onestopenglish'
    onestopenglish_link = 'https://github.com/nishkalavallabhi/OneStopEnglishCorpus/'

    if not os.path.isdir(onestopenglish_path):
      os.mkdir(onestopenglish_path)
      os.chdir(onestopenglish_path)
      clone = 'git clone ' + onestopenglish_link
      os.system(clone)
      
      onestopenglish_files = sorted(glob.glob(f"{onestopenglish_path}/OneStopEnglishCorpus/Texts-Together-OneCSVperFile/*"))

      src_ids = []
      src = []
      simp_ids = []
      simp = []
      topics = []
      origin = []
      duplicated = []
      curr_src_id = -1

      for onestopenglish_file in onestopenglish_files:
        onestopenglish_simp_df = pd.read_csv(onestopenglish_file, encoding='latin1', header=0)
        onestopenglish_simp_df.dropna(inplace=True)
        onestopenglish_simp_df = onestopenglish_simp_df.reset_index()

        topic = onestopenglish_file[len(onestopenglish_path) + 76:-4]

        if 'Intermediate' in onestopenglish_simp_df.columns:
          intermediate = 'Intermediate'
        else:
          intermediate = 'Intermediate '

        for index, row in onestopenglish_simp_df.iterrows():
          ticks = 0
          if row['Advanced'] != row[intermediate]:
            duplicated.append(False)
          else:
            duplicated.append(True)
          
          src_ids.append(curr_src_id + 1)
          ticks = ticks + 0.5
          src.append(row['Advanced'])

          simp_ids.append(len(simp_ids))
          simp.append(row[intermediate])

          topics.append(topic)
          origin.append('advanced-intermediate')

          curr_simp_id = len(simp_ids)
          
          if row['Advanced'] != row['Elementary']:
            duplicated.append(False)
          else:
            duplicated.append(True)
          
          src_ids.append(curr_src_id + 1)
          ticks = ticks + 0.5
          src.append(row['Advanced'])

          simp_ids.append(curr_simp_id)
          simp.append(row['Elementary'])

          topics.append(topic)
          origin.append('advanced-elementary')

          if row[intermediate] != row['Elementary']:
            duplicated.append(False)
          else:
            duplicated.append(True)
          
          src_ids.append(curr_src_id + 2)
          ticks = ticks + 1
          src.append(row[intermediate])
          
          simp_ids.append(curr_simp_id)
          simp.append(row['Elementary'])

          topics.append(topic)
          origin.append('intermediate-elementary')

          curr_src_id = curr_src_id + floor(ticks)

      full_data = {'ds_id': 'OneStopEnglish', 'src': src, 'src_id': src_ids, 'simp': simp, 'simp_id': simp_ids, 'origin': origin, 'topic': topics, 'granularity': 'sentence', 'duplicated': duplicated}
      onestopenglish_dataset = pd.DataFrame(data = full_data)

      with open('/' + onestopenglish_path + '/onestopenglish.pkl', 'wb') as f:
        pickle.dump(onestopenglish_dataset, f)
  else:
    onestopenglish_dataset = pd.read_pickle(path_to_datasets + 'onestopenglish/onestopenglish.pkl')
  return onestopenglish_dataset

def load_pwkp_ds():
  # PWKP dataset
  # https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2447

  if not os.path.isfile(path_to_datasets + 'pwkp/pwkp.pkl'): 
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
      duplicated = []
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
              if source_sent != simp_sents:
                duplicated.append(False)
              else:
                duplicated.append(True)
              
              simp.append(simp_sents)
              src.append(source_sent)

              src_ids.append(curr_src_id)
              simp_ids.append(len(simp_ids))

            last_line_blank = True
            simp_sents = ''
            source_sent = ''

      full_data = {'ds_id': 'PWKP', 'src': src, 'src_id': src_ids, 'simp': simp, 'simp_id': simp_ids, 'granularity': 'sentence', 'duplicated': duplicated}
      pwkp_dataset = pd.DataFrame(data = full_data)

      with open(pwkp_path + '/pwkp.pkl', 'wb') as f:
        pickle.dump(pwkp_dataset, f)
  else:
    pwkp_dataset = pd.read_pickle(path_to_datasets + 'pwkp/pwkp.pkl')
  
  return pwkp_dataset

def load_questeval_ds():
  # QuestEval dataset
  # http://dl.fbaipublicfiles.com/questeval/simplification_human_evaluations.tar.gz

  if not os.path.isfile(path_to_datasets + 'questeval/questeval.pkl'): 
    questeval_path = path_to_datasets + 'questeval'
    questeval_link = 'http://dl.fbaipublicfiles.com/questeval/simplification_human_evaluations.tar.gz'

    if not os.path.isdir(questeval_path):
      os.mkdir(questeval_path)
      
      response = requests.get(questeval_link, stream=True)
      
      if response.status_code == 200:
        with open('/' + questeval_path + '/data.tar.gz', 'wb') as f:
            f.write(response.raw.read())

        f = tarfile.open('/' + questeval_path + '/data.tar.gz')
        f.extractall(questeval_path)
        f.close()

        src_ids = []
        src = []
        simp_ids = []
        simp = []
        duplicated = []
        
        src_ids_hlp = {}
        src_info = set()

        questeval_df = pd.read_csv(questeval_path + '/simplification_human_evaluations/questeval_simplification_likert_ratings.csv', encoding='latin1', header=0)
        for index, row in questeval_df.iterrows():
          src_hlp = row['source']
          simp_hlp = row['simplification']
          comb = src_hlp + '$$$$$' + simp_hlp

          if comb not in src_info:
            src_info.add(comb)

            if src_hlp not in src_ids_hlp:
              src_ids_hlp[src_hlp] = len(src_ids_hlp)

            src_ids.append(src_ids_hlp[src_hlp])
            src.append(src_hlp)

            simp_ids.append(len(simp_ids))
            simp.append(simp_hlp)

            if src_hlp == simp_hlp:
              duplicated.append(True)
            else:
              duplicated.append(False)

        full_data = {'ds_id': 'QuestEval', 'src': src, 'src_id': src_ids, 'simp': simp, 'simp_id': simp_ids, 'granularity': 'sentence', 'duplicated': duplicated}
        questeval_dataset = pd.DataFrame(data = full_data)

        with open(questeval_path + '/questeval.pkl', 'wb') as f:
          pickle.dump(questeval_dataset, f)
  else:
    questeval_dataset = pd.read_pickle(path_to_datasets + 'questeval/questeval.pkl')
  
  return questeval_dataset       

def load_semeval07_ds():
  # SemEval_2007 dataset
  # http://www.dianamccarthy.co.uk/files/task10data.tar.gz

  if not os.path.isfile(path_to_datasets + 'semeval07/semeval07.pkl'): 
    semeval07_path = path_to_datasets + 'semeval07'
    semeval07_link = 'http://www.dianamccarthy.co.uk/files/task10data.tar.gz'

    if not os.path.isdir(semeval07_path):
      os.mkdir(semeval07_path)
      
      response = requests.get(semeval07_link, stream=True)
      
      if response.status_code == 200:
        with open('/' + semeval07_path + '/data.tar.gz', 'wb') as f:
            f.write(response.raw.read())

        f = tarfile.open('/' + semeval07_path + '/data.tar.gz')
        f.extractall(semeval07_path)
        f.close()

        src_ids = []
        src = []
        simp_ids = []
        simp = []
        duplicated = []
        
        src_info = {}

        with open('/' + semeval07_path + '/trial/lexsub_trial.xml') as fp:
          soup = BeautifulSoup(fp, 'xml')

          lexelts = soup.find_all('lexelt')
          for l in lexelts:
            item = l.get('item')
            instances = l.find_all('instance')

            for i in instances:
              context = i.find('context')
              context_text = str(context).replace(' .', '.').replace(' ,', ',').replace('( ', '(').replace(' )', ')').replace(' ;', ';').replace(' :', ':').replace(' ?', '?').replace(' !', '!').replace('<context>', '').replace('</context>', '').replace('<head>', '').replace('</head>', '')
              context_raw = str(context).replace(' .', '.').replace(' ,', ',').replace('( ', '(').replace(' )', ')').replace(' ;', ';').replace(' :', ':').replace(' ?', '?').replace(' !', '!').replace('<context>', '').replace('</context>', '')
              curr_id = i.get('id')

              left_side = context_raw.split('<head>')[0]
              right_side = context_raw.split('<head>')[1].split('</head>')[1]

              src_info[item + ' ' + curr_id] = {'src_txt': context_text, 'left': left_side, 'right': right_side, 'id': len(src_info)}

        for a in ['/trial/gold.trial', '/trial/mwgold.trial']:
          with open('/' + semeval07_path + a) as fp:
            for l in fp.readlines():
              pts = l.split(' :: ')
              curr_id = pts[0]

              if len(pts) == 2:
                replacements = pts[1].split(';')

                for replacement in replacements[:-1]:
                  src_hlp = src_info[curr_id]['src_txt'].strip()
                  simp_hlp = src_info[curr_id]['left'].strip() + ' ' + replacement[:-1].strip() + ' ' + src_info[curr_id]['right'].replace('\n', '').strip()

                  if src_hlp != simp_hlp:
                    duplicated.append(False)
                  else:
                    duplicated.append(True)

                  src_ids.append(src_info[curr_id]['id'])
                  src.append(src_hlp)

                  simp_ids.append(len(simp_ids))
                  simp.append(simp_hlp)

        full_data = {'ds_id': 'SemEval_2007', 'src': src, 'src_id': src_ids, 'simp': simp, 'simp_id': simp_ids, 'label': 'trial', 'granularity': 'sentence', 'duplicated': duplicated}
        semeval07_dataset = pd.DataFrame(data = full_data)

        with open('/' + semeval07_path + '/semeval07.pkl', 'wb') as f:
          pickle.dump(semeval07_dataset, f)
  else:
    semeval07_dataset = pd.read_pickle(path_to_datasets + 'semeval07/semeval07.pkl')

  return semeval07_dataset

def load_simpa_ds():
  # SIMPA DATASET
  # https://github.com/simpaticoproject/simpa

  if not os.path.isfile(path_to_datasets + 'simpa/simpa.pkl'):
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

      duplicated = []
      for i in range(len(simp)):
        if simp[i] == src[i]:
          duplicated.append(True)
        else: 
          duplicated.append(False)

      full_data = {'ds_id': 'simpa', 'src': src, 'src_id': src_ids, 'simp': simp, 'simp_id': simp_ids, 'origin': origin, 'granularity': 'sentence', 'duplicated': duplicated}
      simpa_dataset = pd.DataFrame(data = full_data)

      with open('/' + simpa_path + '/simpa.pkl', 'wb') as f:
        pickle.dump(simpa_dataset, f)
  else:
    simpa_dataset = pd.read_pickle(path_to_datasets + 'simpa/simpa.pkl')

  return simpa_dataset

def load_simpeval_ds():
  # SimpEval 2022 dataset
  # https://github.com/Yao-Dou/LENS/

  if not os.path.isfile(path_to_datasets + 'simpeval/simpeval.pkl'):
    simpeval_path = path_to_datasets + 'simpeval'
    simpeval_link = 'https://github.com/Yao-Dou/LENS/'

    if not os.path.isdir(simpeval_path):
      os.mkdir(simpeval_path)
      os.chdir(simpeval_path)
      clone = 'git clone ' + simpeval_link
      os.system(clone)

      simpeval_files = sorted(glob.glob(f"{simpeval_path}/LENS/data/*"))

      src_ids = []
      src = []

      simp_ids = []
      simp = []

      origin = []
      duplicated = []

      all_combs = set()

      for simpeval_file in simpeval_files:
        simpeval_df = pd.read_csv(simpeval_file, encoding='latin1', header=0)
    
        if 'Input.original' in simpeval_df.columns:
          input_col = 'Input.original'
        else:
          input_col = 'original'

        if 'Input.simplified' in simpeval_df.columns:
          simplified_col = 'Input.simplified'
        else:
          simplified_col = 'generation'

        for index, row in simpeval_df.iterrows():
          comb = str(row[input_col]) + '$$$$$' + str(row[simplified_col])

          if comb not in all_combs:
            all_combs.add(comb)

            if len(src_ids) == 0 or src[-1] != row[input_col]:
              if len(src_ids) == 0:
                src_ids.append(0)
              else:
                src_ids.append(src_ids[-1] + 1)
              src.append(str(row[input_col]))

              simp_ids.append(len(simp_ids))
              simp.append(str(row[simplified_col]))

              origin.append(str(simpeval_file[len(simpeval_path + '/LENS/data/'):-4]))

              if row[input_col] != row[simplified_col]:
                duplicated.append(False)
              else:
                duplicated.append(True)
            elif src[-1] == row[input_col] and simp[-1] != row[simplified_col]:
              src_ids.append(src_ids[-1])
              src.append(str(row[input_col]))

              simp_ids.append(len(simp_ids))
              simp.append(str(row[simplified_col]))

              origin.append(str(simpeval_file[len(simpeval_path + '/LENS/data/'):-4]))
            
              if row[input_col] != row[simplified_col]:
                duplicated.append(False)
              else:
                duplicated.append(True)

      full_data = {'ds_id': 'SimpEval_22', 'src': src, 'src_id': src_ids, 'simp': simp, 'simp_id': simp_ids, 'origin': origin, 'granularity': 'sentence', 'duplicated': duplicated}
      simpeval_dataset = pd.DataFrame(data = full_data)

      with open('/' + simpeval_path + '/simpeval.pkl', 'wb') as f:
        pickle.dump(simpeval_dataset, f)
  else:
    simpeval_dataset = pd.read_pickle(path_to_datasets + 'simpeval/simpeval.pkl')
  return simpeval_dataset

def load_sscorpus_ds():
  # SSCORPUS dataset
  # https://github.com/tmu-nlp/sscorpus

  if not os.path.isfile(path_to_datasets + 'sscorpus/sscorpus.pkl'):
    sscorpus_path = path_to_datasets + 'sscorpus'
    sscorpus_link = 'https://github.com/tmu-nlp/sscorpus'

    if not os.path.isdir(sscorpus_path):
      os.mkdir(sscorpus_path)
      os.chdir(sscorpus_path)
      clone = 'git clone ' + sscorpus_link
      os.system(clone)

      fp = open(sscorpus_path + '/sscorpus.txt', 'wb')
      with gzip.open(sscorpus_path + '/sscorpus/sscorpus.gz', 'rb') as f:
        bindata = f.read()
        fp.write(bindata)
        fp.close()

      src_ids = []
      src = []
      simp_ids = []
      simp = []
      similarity = []
      duplicated = []

      with open('/' + sscorpus_path + '/sscorpus.txt') as f:
        lines = f.readlines()
        # complex sentence TAB simple sentence TAB similarity score
        for l in lines:
          parts = l.split('\t')

          src_hlp = parts[0].replace(' .', '.').replace(' ,', ',').replace('( ', '(').replace(' )', ')').replace(' ;', ';').replace(' :', ':').replace(" '", "'").replace("` ", "`")
          simp_hlp = parts[1].replace(' .', '.').replace(' ,', ',').replace('( ', '(').replace(' )', ')').replace(' ;', ';').replace(' :', ':').replace(" '", "'").replace("` ", "`")
          
          if src_hlp != simp_hlp:
            duplicated.append(False)
          else:
            duplicated.append(True)

          src.append(src_hlp)
          simp.append(simp_hlp)
          similarity.append(float(parts[2]))
          src_ids.append(len(src_ids))
          simp_ids.append(len(simp_ids))

      full_data = {'ds_id': 'SSCORPUS', 'src': src, 'src_id': src_ids, 'simp': simp, 'simp_id': simp_ids, 'similarity': similarity, 'granularity': 'sentence', 'duplicated': duplicated}
      sscorpus_dataset = pd.DataFrame(data = full_data)

      with open('/' + sscorpus_path + '/sscorpus.pkl', 'wb') as f:
        pickle.dump(sscorpus_dataset, f)
  else:
    sscorpus_dataset = pd.read_pickle(path_to_datasets + 'sscorpus/sscorpus.pkl')
  return sscorpus_dataset

def load_turkcorpus_ds():
  # TurkCorpus dataset
  # https://github.com/cocoxu/simplification/tree/master

  if not os.path.isfile(path_to_datasets + 'turkcorpus/turkcorpus.pkl'):
    turkcorpus_path = path_to_datasets + 'turkcorpus'
    turkcorpus_link = 'https://github.com/cocoxu/simplification'

    if not os.path.isdir(turkcorpus_path):
      os.mkdir(turkcorpus_path)
      os.chdir(turkcorpus_path)
      clone = 'git clone ' + turkcorpus_link
      os.system(clone)

      turkcorpus_files = sorted(glob.glob(f"{turkcorpus_path}/simplification/data/turkcorpus/GEM/*"))

      src_ids = []
      src = []
      simp_ids = []
      simp = []
      origin = []
      label = []
      duplicated = []

      with open('/' + turkcorpus_path + '/simplification/data/turkcorpus/test.8turkers.tok.simp') as f:
        lines = f.readlines()
        for l in lines:
          simp_ids.append(len(simp_ids))
          simp.append(l.replace(' .', '.').replace(' ,', ',').replace('( ', '(').replace(' )', ')').replace(' ;', ';').replace(' :', ':'))
          label.append('test')
          origin.append('Simple_Wikipedia')

      with open('/' + turkcorpus_path + '/simplification/data/turkcorpus/tune.8turkers.tok.simp') as f:
        lines = f.readlines()
        for l in lines:
          simp_ids.append(len(simp_ids))
          simp.append(l.replace(' .', '.').replace(' ,', ',').replace('( ', '(').replace(' )', ')').replace(' ;', ';').replace(' :', ':'))
          label.append('tune')
          origin.append('Simple_Wikipedia')

      for path in turkcorpus_files:
        with open('/' + path) as f:
          lines = f.readlines()

          # Wikipedia sentences
          if path[-4:] == 'norm':
            for l in lines:
              src_ids.append(len(src_ids))
              src.append(l.replace(' .', '.').replace(' ,', ',').replace('( ', '(').replace(' )', ')').replace(' ;', ';').replace(' :', ':'))

          # Turker sentences
          else:
            if path[-24:-20] == 'tune':
              curr_label = 'tune'
            else:
              curr_label = 'test'

            for l in lines:
              simp_ids.append(len(simp_ids))
              simp.append(l.replace(' .', '.').replace(' ,', ',').replace('( ', '(').replace(' )', ')').replace(' ;', ';').replace(' :', ':'))
              label.append(curr_label)
              origin.append('Turker_' + path[-1])

      multiplicator = 9
      src = src * multiplicator
      src_ids = src_ids * multiplicator

      for i in range(len(simp)):
        if simp[i] == src[i]:
          duplicated.append(True)
        else: 
          duplicated.append(False)

      full_data = {'ds_id': 'TurkCorpus', 'src': src, 'src_id': src_ids, 'simp': simp, 'simp_id': simp_ids, 'label': label, 'origin': origin, 'granularity': 'sentence', 'duplicated': duplicated}
      turkcorpus_dataset = pd.DataFrame(data = full_data)

      with open('/' + turkcorpus_path + '/turkcorpus.pkl', 'wb') as f:
        pickle.dump(turkcorpus_dataset, f)
  else:
    turkcorpus_dataset = pd.read_pickle(path_to_datasets + 'turkcorpus/turkcorpus.pkl')
  return turkcorpus_dataset
  
def load_wikiauto_ds():
  # Wiki-Auto (ACL) dataset
  # https://github.com/chaojiang06/wiki-auto

  if not os.path.isfile(path_to_datasets + 'wikiauto/wikiauto.pkl'):
    wikiauto_path = path_to_datasets + 'wikiauto'
    wikiauto_link = 'https://github.com/chaojiang06/wiki-auto'

    if not os.path.isdir(wikiauto_path):
      os.mkdir(wikiauto_path)
      os.chdir(wikiauto_path)
      clone = 'git clone ' + wikiauto_link
      os.system(clone)

      src_ids = []
      src = []
      simp_ids = []
      simp = []
      duplicated = []

      with open('/' + wikiauto_path + '/wiki-auto/wiki-auto/ACL2020/train.src') as f:
        lines = f.readlines()
        for l in lines:
          src_ids.append(len(src_ids))
          src.append(l.replace(' -RRB- ', ' ').replace(' -LRB- ', ' ').replace(' -RSB- ', ' ').replace(' .', '.').replace(' ,', ',').replace('( ', '(').replace(' )', ')').replace(' ;', ';').replace(' :', ':'))
      
      with open('/' + wikiauto_path + '/wiki-auto/wiki-auto/ACL2020/train.dst') as f:
        lines = f.readlines()
        for l in lines:
          simp_ids.append(len(simp_ids))
          simp.append(l.replace(' -RRB- ', ' ').replace(' -LRB- ', ' ').replace(' -RSB- ', ' ').replace(' .', '.').replace(' ,', ',').replace('( ', '(').replace(' )', ')').replace(' ;', ';').replace(' :', ':'))

      for i in range(len(simp)):
        if simp[i] == src[i]:
          duplicated.append(True)
        else: 
          duplicated.append(False)

      full_data = {'ds_id': 'Wiki-Auto', 'src': src, 'src_id': src_ids, 'simp': simp, 'simp_id': simp_ids, 'label': 'train', 'granularity': 'sentence', 'duplicated': duplicated}
      wikiauto_dataset = pd.DataFrame(data = full_data)

      with open('/' + wikiauto_path + '/wikiauto.pkl', 'wb') as f:
        pickle.dump(wikiauto_dataset, f)
  else:
    wikiauto_dataset = pd.read_pickle(path_to_datasets + 'wikiauto/wikiauto.pkl')
  return wikiauto_dataset

def load_wikimanual_ds():
  # Wiki-Manual dataset, aligned, non-duplicates
  # https://github.com/chaojiang06/wiki-auto
  # and https://www.dropbox.com/sh/ohqaw41v48c7e5p/AACdl4UPKtu7CMMa-CJhz4G7a/wiki-manual/train.tsv?dl=0 (to download manually and put into datasets folder)

  if not os.path.isfile(path_to_datasets + 'wikimanual/wikimanual.pkl'):
    wikimanual_path = path_to_datasets + 'wikimanual'
    wikimanual_link = 'https://github.com/chaojiang06/wiki-auto'

    if not os.path.isdir(wikimanual_path):
      os.mkdir(wikimanual_path)
      os.chdir(wikimanual_path)
      clone = 'git clone ' + wikimanual_link
      os.system(clone)     

      src_ids = []
      src = []
      simp_ids = []
      simp = []
      label = []
      duplicated = []
      srcs = {}

      with open('/' + wikimanual_path + '/wiki-auto/wiki-manual/dev.tsv') as f:
        lines = f.readlines()
        for l in lines:
          pts = l.split('\t')
          if pts[0] == 'aligned':
            if pts[4] not in srcs:
              srcs[pts[4]] = len(srcs)

            src_ids.append(srcs[pts[4]])
            src.append(pts[4])
            simp_ids.append(len(src_ids))
            simp.append(pts[3])
            label.append('dev')

            if pts[3] != pts[4]:
              duplicated.append(False)
            else:
              duplicated.append(True)

      with open('/' + wikimanual_path + '/wiki-auto/wiki-manual/test.tsv') as f:
        lines = f.readlines()
        for l in lines:
          pts = l.split('\t')
          if pts[0] == 'aligned':
            if pts[4] not in srcs:
              srcs[pts[4]] = len(srcs)

            src_ids.append(srcs[pts[4]])
            src.append(pts[4])
            simp_ids.append(len(src_ids))
            simp.append(pts[3])
            label.append('test')

            if pts[3] != pts[4]:
              duplicated.append(False)
            else:
              duplicated.append(True)

      if os.path.exists('/' + path_to_datasets + '/train.tsv'):
        with open('/' + path_to_datasets + '/train.tsv') as f:
          lines = f.readlines()
          for l in lines:
            pts = l.split('\t')
            if pts[0] == 'aligned':
              if pts[4] not in srcs:
                srcs[pts[4]] = len(srcs)

              src_ids.append(srcs[pts[4]])
              src.append(pts[4])
              simp_ids.append(len(src_ids))
              simp.append(pts[3])
              label.append('train')

              if pts[3] != pts[4]:
                duplicated.append(False)
              else:
                duplicated.append(True)
      else:
        print('Please download the Train split for the Wiki-Manual dataset from Dropbox and put it into the datasets folder: https://www.dropbox.com/sh/ohqaw41v48c7e5p/AACdl4UPKtu7CMMa-CJhz4G7a/wiki-manual/train.tsv?dl=0')

      full_data = {'ds_id': 'Wiki-Manual', 'src': src, 'src_id': src_ids, 'simp': simp, 'simp_id': simp_ids, 'label': label, 'granularity': 'sentence', 'duplicated': duplicated}
      wikimanual_dataset = pd.DataFrame(data = full_data)

      with open('/' + wikimanual_path + '/wikimanual.pkl', 'wb') as f:
        pickle.dump(wikimanual_dataset, f)
  else:
    wikimanual_dataset = pd.read_pickle(path_to_datasets + 'wikimanual/wikimanual.pkl')
  return wikimanual_dataset

def load_wikisplit_ds():
  # WikiSplit dataset
  # https://github.com/google-research-datasets/wiki-split

  if not os.path.isfile(path_to_datasets + 'wikisplit/wikisplit.pkl'):
    wikisplit_path = path_to_datasets + 'wikisplit'
    wikisplit_link = 'https://github.com/google-research-datasets/wiki-split'

    if not os.path.isdir(wikisplit_path):
      os.mkdir(wikisplit_path)
      os.chdir(wikisplit_path)
      clone = 'git clone ' + wikisplit_link
      os.system(clone)

      with zipfile.ZipFile(wikisplit_path + '/wiki-split/train.tsv.zip', 'r') as f:
          f.extractall(wikisplit_path + '/wiki-split/')

      src_ids = []
      src = []
      simp_ids = []
      simp = []
      labels = []
      duplicated = []

      wikisplit_files = sorted(glob.glob(f"{wikisplit_path}/wiki-split/*"))

      for wikisplit_file in wikisplit_files:
        if wikisplit_file[-3:] == 'tsv':
          with open('/' + wikisplit_file) as f:
            lines = f.readlines()
            for l in lines:
              pts = l.split('\t')

              src_hlp = pts[0].replace(' ,', ',').replace(' .', '.').replace(' :', ':').replace(' ;', ';').replace(' ?', '?').replace(' !', '!').replace('( ', '(').replace(' )', ')')
              simp_hlp = pts[1].replace(' <::::> ', ' ').replace(' ,', ',').replace(' .', '.').replace(' :', ':').replace(' ;', ';').replace(' ?', '?').replace(' !', '!').replace('( ', '(').replace(' )', ')')

              if src_hlp != simp_hlp:
                duplicated.append(False)
              else:
                duplicated.append(True)

              src_ids.append(len(src_ids))
              src.append(src_hlp)
              simp_ids.append(len(simp_ids))
              simp.append(simp_hlp)
              labels.append(wikisplit_file[len(wikisplit_path + '/wiki-split/'):-4])

      full_data = {'ds_id': 'WikiSplit', 'src': src, 'src_id': src_ids, 'simp': simp, 'simp_id': simp_ids, 'label': labels, 'granularity': 'sentence', 'duplicated': duplicated}
      wikisplit_dataset = pd.DataFrame(data=full_data)

      with open('/' + wikisplit_path + '/wikisplit.pkl', 'wb') as f:
        pickle.dump(wikisplit_dataset, f)
  else:
    wikisplit_dataset = pd.read_pickle(path_to_datasets + 'wikisplit/wikisplit.pkl')

  return wikisplit_dataset

def load_wikipediav1_ds():
  # Wikipedia version 1 dataset
  # https://cs.pomona.edu/~dkauchak/simplification/data.v1/data.v1.split.tar.gz

  if not os.path.isfile(path_to_datasets + 'wikipediav1/wikipediav1.pkl'): 
    wikipediav1_path = path_to_datasets + 'wikipediav1'
    wikipediav1_link = 'https://cs.pomona.edu/~dkauchak/simplification/data.v1/data.v1.split.tar.gz'

    if not os.path.isdir(wikipediav1_path):
      os.mkdir(wikipediav1_path)
      
    response = requests.get(wikipediav1_link, stream=True)
    
    if response.status_code == 200:
      with open('/' + wikipediav1_path + '/data.tar.gz', 'wb') as f:
          f.write(response.raw.read())

      f = tarfile.open(wikipediav1_path + '/data.tar.gz')
      f.extractall(wikipediav1_path)
      f.close()

      src_ids = []
      src = []
      simp_ids = []
      simp = []
      labels = []
      duplicated = []

      wikipediav1_files = sorted(glob.glob(f"{wikipediav1_path}/data.v1.split/*"))
      for wikipediav1_file in wikipediav1_files:
        if wikipediav1_file[-3:] == 'txt':
          is_simp = wikipediav1_file[len(wikipediav1_path + '/data.v1.split/'):].split('.')[0] == 'simple'
          file_type = wikipediav1_file[len(wikipediav1_path + '/data.v1.split/'):].split('.')[1]

          with open('/' + wikipediav1_file) as f:
            lines = f.readlines()

            for l in lines:
              if is_simp:
                simp_ids.append(len(simp_ids))
                simp.append(l.replace(' ,', ',').replace(' .', '.').replace(' :', ':').replace(' ;', ';').replace(' ?', '?').replace(' !', '!').replace('( ', '(').replace(' )', ')'))
                labels.append(file_type)
              else:
                src_ids.append(len(src_ids))
                src.append(l.replace(' ,', ',').replace(' .', '.').replace(' :', ':').replace(' ;', ';').replace(' ?', '?').replace(' !', '!').replace('( ', '(').replace(' )', ')'))

      for i in range(len(simp)):
        if simp[i] == src[i]:
          duplicated.append(True)
        else: 
          duplicated.append(False)

      full_data = {'ds_id': 'Wikipedia_v1', 'src': src, 'src_id': src_ids, 'simp': simp, 'simp_id': simp_ids, 'label': labels, 'granularity': 'sentence', 'duplicated': duplicated}
      wikipediav1_dataset = pd.DataFrame(data=full_data)

      with open('/' + wikipediav1_path + '/wikipediav1.pkl', 'wb') as f:
        pickle.dump(wikipediav1_dataset, f)
  else:
    wikipediav1_dataset = pd.read_pickle(path_to_datasets + 'wikipediav1/wikipediav1.pkl')

  return wikipediav1_dataset

def load_wikipediav2_ds():
  # Wikipedia version 2 dataset
  # https://cs.pomona.edu/~dkauchak/simplification/data.v2/sentence-aligned.v2.tar.gz
  # and https://cs.pomona.edu/~dkauchak/simplification/data.v2/document-aligned.v2.tar.gz

  if not os.path.isfile(path_to_datasets + 'wikipediav2/wikipediav2.pkl'): 
    wikipediav2_path = path_to_datasets + 'wikipediav2'
    wikipediav2_link_sent = 'https://cs.pomona.edu/~dkauchak/simplification/data.v2/sentence-aligned.v2.tar.gz'
    wikipediav2_link_doc = 'https://cs.pomona.edu/~dkauchak/simplification/data.v2/document-aligned.v2.tar.gz'

    if not os.path.isdir(wikipediav2_path):
      os.mkdir(wikipediav2_path)
      
    response = requests.get(wikipediav2_link_sent, stream=True)
    response2 = requests.get(wikipediav2_link_doc, stream=True)

    if response.status_code == 200 and response2.status_code == 200:
      with open('/' + wikipediav2_path + '/data_sent.tar.gz', 'wb') as f:
          f.write(response.raw.read())

      f = tarfile.open(wikipediav2_path + '/data_sent.tar.gz')
      f.extractall(wikipediav2_path)
      f.close()

      with open('/' + wikipediav2_path + '/data_doc.tar.gz', 'wb') as f2:
          f2.write(response2.raw.read())

      f2 = tarfile.open(wikipediav2_path + '/data_doc.tar.gz')
      f2.extractall(wikipediav2_path)
      f2.close()

      src_ids = []
      src = []
      simp_ids = []
      simp = []
      topics = []
      granularities = []
      duplicated = []

      all_src = []
      all_simp = []
      all_topics = []

      wikipediav2_files = sorted(glob.glob(f"{wikipediav2_path}/sentence-aligned.v2/*"))
      for wikipediav2_file in wikipediav2_files:
        if wikipediav2_file[-7:] == 'aligned':
          is_simp = wikipediav2_file[len(wikipediav2_path + '/sentence-aligned.v2/'):].split('.')[0] == 'simple'

          with open('/' + wikipediav2_file) as f:
            lines = f.readlines()

            for l in lines:
              pts = l.split('\t')

              if is_simp:
                all_simp.append(pts[2].replace(' ,', ',').replace(' .', '.').replace(' :', ':').replace(' ;', ';').replace(' ?', '?').replace(' !', '!').replace('( ', '(').replace(' )', ')'))
              else:
                all_src.append(pts[2].replace(' ,', ',').replace(' .', '.').replace(' :', ':').replace(' ;', ';').replace(' ?', '?').replace(' !', '!').replace('( ', '(').replace(' )', ')'))
                all_topics.append(pts[0])

      all_doc_simp = {}
      all_doc_src = {}

      wikipediav2_files = sorted(glob.glob(f"{wikipediav2_path}/document-aligned.v2/*"))
      for wikipediav2_file in wikipediav2_files:
        if wikipediav2_file[-3:] == 'txt':
          is_simp = wikipediav2_file[len(wikipediav2_path + '/document-aligned.v2/'):].split('.')[0] == 'simple'

          with open('/' + wikipediav2_file) as f:
            lines = f.readlines()

            curr_topic = ''
            curr_doc = ''
            curr_para = 0
            for l in lines:
              pts = l.split('\t')

              if pts[0] == curr_topic:
                if curr_para == int(pts[1]):
                  curr_doc = curr_doc + ' ' + pts[2]
                else:
                  curr_doc = curr_doc + '\n' + pts[2]
                  curr_para = int(pts[1])
              else:
                if len(curr_doc) > 0:
                  txt = curr_doc.replace(' ,', ',').replace(' .', '.').replace(' :', ':').replace(' ;', ';').replace(' ?', '?').replace(' !', '!').replace('( ', '(').replace(' )', ')')

                  if is_simp:
                    all_doc_simp[curr_topic] = txt
                  else:
                    all_doc_src[curr_topic] = txt

                curr_doc = pts[2]
                curr_topic = pts[0]
                curr_para = int(pts[1])

            if len(curr_doc) > 0:
              txt = curr_doc.replace(' ,', ',').replace(' .', '.').replace(' :', ':').replace(' ;', ';').replace(' ?', '?').replace(' !', '!').replace('( ', '(').replace(' )', ')')

              if is_simp:
                all_doc_simp[curr_topic] = txt
              else:
                all_doc_src[curr_topic] = txt

      for i in range(len(all_src)):
        if all_src[i] != all_simp[i]:
          duplicated.append(False)
        else: 
          duplicated.append(True)

        src_ids.append(len(src_ids))
        src.append(all_src[i])
        simp_ids.append(len(simp_ids))
        simp.append(all_simp[i])
        topics.append(all_topics[i])
        granularities.append('sentence')
      
      for topic in all_doc_src:
        if topic in all_doc_simp:
          src_hlp = all_doc_src[topic]
          simp_hlp = all_doc_simp[topic]
          if src_hlp != simp_hlp:
            duplicated.append(False)
          else: 
            duplicated.append(True)
          src_ids.append(len(src_ids))
          src.append(src_hlp)
          simp_ids.append(len(simp_ids))
          simp.append(simp_hlp)
          topics.append(topic)
          granularities.append('document')
      
      full_data = {'ds_id': 'Wikipedia_v2', 'src': src, 'src_id': src_ids, 'simp': simp, 'simp_id': simp_ids, 'topics': topics, 'granularity': granularities, 'duplicated': duplicated}
      wikipediav2_dataset = pd.DataFrame(data=full_data)

      with open('/' + wikipediav2_path + '/wikipediav2.pkl', 'wb') as f:
        pickle.dump(wikipediav2_dataset, f)
  else:
    wikipediav2_dataset = pd.read_pickle(path_to_datasets + 'wikipediav2/wikipediav2.pkl')

  return wikipediav2_dataset

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


def add_global_index(df):
  df['index'] = df.apply(lambda row: f"{row['ds_id']}__{row['src_id']}__{row['simp_id']}", axis=1)
  with open("/workspace/datasets/final_combined_with_index.pkl", "wb") as out_f:
        pickle.dump(df, out_f)

  return df
