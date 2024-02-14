import numpy as np
import pandas as pd

import warnings

from snorkel.labeling import labeling_function
from snorkel.labeling import LabelingFunction
from snorkel.preprocess import preprocessor

from wordfreq import word_frequency

import spacy
from spacy_syllables import SpacySyllables
import spacy_universal_sentence_encoder

from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging

import textstat
from PassivePySrc import PassivePy
from Levenshtein import distance

import language_tool_python
passivepy = PassivePy.PassivePyAnalyzer(spacy_model = "en_core_web_sm")

from qanom.nominalization_detector import NominalizationDetector
nom_detector = NominalizationDetector()

ABSTAIN = -1
SIMPLE = 0
NOT_SIMPLE = 1
LOST_MEANING = 2

label_map = {5: "ABSTAIN", 0: "SIMPLE", 1: "NOT_SIMPLE", 2: "LOST_MEANING"}

#resources
aoa_dic = None
concreteness_dic = None
imageability_dic = None
predictor = None
tool_us = None
tool_gb = None

def init():
  print("resources get initialised")

  global aoa_dic
  global concreteness_dic 
  global imageability_dic 
  global predictor 
  global tool_us 
  global tool_gb
  global ox5k_a
  global academic_word_list

  aoa_list = pd.read_excel("/workspace/datasets/other_resources/AoA_ratings_Kuperman_et_al_BRM.xlsx")
  aoa_list = aoa_list.drop(["OccurTotal", "OccurNum", "Freq_pm", "Rating.SD", "Dunno"], axis=1)
  aoa_list = aoa_list.set_index('Word')
  aoa_dic = aoa_list['Rating.Mean'].to_dict()

  concrete_list = pd.read_excel("/workspace/datasets/other_resources/13428_2013_403_MOESM1_ESM.xlsx")
  concrete_list = concrete_list.drop(["Bigram", "Conc.SD", "Unknown", "Total", "Percent_known", "SUBTLEX"], axis=1)
  concrete_list = concrete_list.set_index('Word')
  concreteness_dic = concrete_list['Conc.M'].to_dict()

  imageability_df = pd.read_csv("/workspace/datasets/other_resources/megahr.en", delimiter="\t", names=["Word", "concreteness", "Imageability"])
  imageability_df = imageability_df.drop(['concreteness'], axis=1).set_index('Word')
  imageability_dic = imageability_df['Imageability'].to_dict()

  model_url = "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz"
  predictor = Predictor.from_path(model_url)

  tool_us = language_tool_python.LanguageTool('en-US')
  tool_gb = language_tool_python.LanguageTool('en-GB')

  df_ox5k = pd.read_csv("/workspace/datasets/other_resources/oxford-5k.csv")
  ox5k_a = df_ox5k.loc[df_ox5k["level"].isin(['a1','a2'])]["word"].to_list()

  academic_word_list = [line[:-1] for line in open("/workspace/datasets/other_resources/academic_word_list.csv", "r")]

init()

#preprocessors
def entities_in_list_of_tokens(l_tokens):
  entities = []
  for i, a in enumerate(l_tokens):
    if a.ent_iob_ == "B":
      s = a.text
      t = i
      while len(l_tokens)>t+1 and l_tokens[t+1].ent_iob_ == "I":
        s = s+" "+l_tokens[t+1].text
        t += 1
      entities.append(s)
  return(entities)

def paragraph_sep(doc):
  c_list = []
  f_list = []
  for token in doc:
    if token.tag_ != "_SP":
      c_list.append(token)
    else:
      f_list.append(c_list)
      c_list = [token]
  f_list.append(c_list)
  return(f_list)


@preprocessor(memoize=True)
def spacy_nlp(x):
  nlp = spacy.load('en_core_web_sm')
  nlp.add_pipe("syllables", after="tagger")
  x.pipeline_components = nlp.pipe_names
  x.simp_text = x.simplified_snt
  x.source_text = x.source_snt

  # simplified
  doc = nlp(x.simplified_snt)
  x.simp_syllables = [token._.syllables for token in doc]
  x.simp_syllables_cnt = [token._.syllables_count for token in doc]
  x.simp_tokens = [token.text for token in doc]
  x.simp_tokens_data = [token for token in doc] #token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop
  # list of pos tags: https://melaniewalsh.github.io/Intro-Cultural-Analytics/05-Text-Analysis/13-POS-Keywords.html
  x.simp_words = [token.text for token in doc if token.pos_ != 'PUNCT']
  x.simp_sentences = [s.text for s in doc.sents]
  x.simp_doc = doc
  x.simp_entities = [e.text for e in doc.ents]

  # source
  doc = nlp(x.source_snt)
  x.source_syllables = [token._.syllables for token in doc]
  x.source_syllables_cnt = [token._.syllables_count for token in doc]
  x.source_tokens = [token.text for token in doc]
  x.source_tokens_data = [token for token in doc] #token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop
  # list of pos tags: https://melaniewalsh.github.io/Intro-Cultural-Analytics/05-Text-Analysis/13-POS-Keywords.html
  x.source_words = [token.text for token in doc if token.pos_ != 'PUNCT']
  x.source_sentences = [s.text for s in doc.sents]
  x.source_doc = doc
  x.source_entities = [e.text for e in doc.ents]

  return x

@preprocessor(memoize=True)
def spacy_nlp_paragraph(x):
  nlp = spacy.load('en_core_web_sm')
  nlp.add_pipe("syllables", after="tagger")
  x.pipeline_components = nlp.pipe_names
  x.simp_text = x.simplified_snt
  x.source_text = x.source_snt

  # simplified
  doc = nlp(x.simplified_snt)
  x.simp_syllables = [token._.syllables for token in doc]
  x.simp_syllables_cnt = [token._.syllables_count for token in doc]
  x.simp_tokens = [token.text for token in doc]
  x.simp_tokens_data = [token for token in doc] #token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop
  # list of pos tags: https://melaniewalsh.github.io/Intro-Cultural-Analytics/05-Text-Analysis/13-POS-Keywords.html
  x.simp_words = [token.text for token in doc if token.pos_ != 'PUNCT']
  x.simp_sentences = [s.text for s in doc.sents]
  x.simp_doc = doc
  x.simp_entities = [e.text for e in doc.ents]
  x.simp_paragraph_tokens_data = paragraph_sep(doc)
  # source
  doc = nlp(x.source_snt)
  x.source_syllables = [token._.syllables for token in doc]
  x.source_syllables_cnt = [token._.syllables_count for token in doc]
  x.source_tokens = [token.text for token in doc]
  x.source_tokens_data = [token for token in doc] #token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop
  # list of pos tags: https://melaniewalsh.github.io/Intro-Cultural-Analytics/05-Text-Analysis/13-POS-Keywords.html
  x.source_words = [token.text for token in doc if token.pos_ != 'PUNCT']
  x.source_sentences = [s.text for s in doc.sents]
  x.source_doc = doc
  x.source_entities = [e.text for e in doc.ents]
  x.source_paragraph_tokens_data = paragraph_sep(doc)

  return x



# christin: average Levenshtein distance between original and simplified~\cite{DBLP:conf/acl/NarayanG14}
def avg_Levenshtein(x, lev_threshold, label):
  matched_sentences_source_to_simp = []
  matched_sentences_simp_to_source = []

  max_sent_len = -1
  for source_id in range(len(x.source_sentences)):
    curr_sent_len = len(x.source_sentences[source_id])
    if curr_sent_len > max_sent_len:
      max_sent_len = curr_sent_len

    max_match = -1
    min_lev = len(x.source_text)
    for simp_id in range(len(x.simp_sentences)):
      curr_lev = distance(x.source_sentences[source_id], x.simp_sentences[simp_id])

      if curr_lev < min_lev:
        min_lev = curr_lev
        max_match = simp_id

    if max_match > -1:
      matched_sentences_source_to_simp.append(min_lev)

  for simp_id in range(len(x.simp_sentences)):
    curr_sent_len = len(x.simp_sentences[simp_id])
    if curr_sent_len > max_sent_len:
      max_sent_len = curr_sent_len

    max_match = -1
    min_lev = len(x.simp_text)
    for source_id in range(len(x.source_sentences)):
      curr_lev = distance(x.simp_sentences[simp_id], x.source_sentences[source_id])

      if curr_lev < min_lev:
        min_lev = curr_lev
        max_match = simp_id

    if max_match > -1:
      matched_sentences_simp_to_source.append(min_lev)

  avg_lev = sum(matched_sentences_source_to_simp + matched_sentences_simp_to_source)/(len(matched_sentences_source_to_simp) + len(matched_sentences_simp_to_source))

  if label == SIMPLE:
    if avg_lev <= lev_threshold * max_sent_len:
      return label
    else:
      return ABSTAIN
  else:
    if avg_lev > lev_threshold * max_sent_len:
      return label
    else:
      return ABSTAIN

def low_avg_Levenshtein(lev_threshold, label):
  return LabelingFunction(
      name=f"low_avg_Levenshtein_threshold={lev_threshold}",
      f=avg_Levenshtein,
      resources=dict(lev_threshold=lev_threshold, label=label),
      pre=[spacy_nlp]
  )


# christin: fewer modifiers~\cite{DBLP:conf/acl/NarayanG14}
@labeling_function(pre=[spacy_nlp], name="fewer_modifiers")
def lf_fewer_modifiers(x):
  mods = ['advmod', 'amod', 'nmod', 'npadvmod', 'nummod', 'quantmod']

  deps_source = [token.dep_ for token in x.source_tokens_data if token.dep_ in mods]
  deps_simp = [token.dep_ for token in x.simp_tokens_data if token.dep_ in mods]

  if len(deps_simp) < len(deps_source):
    return SIMPLE

  return ABSTAIN
