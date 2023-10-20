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

@preprocessor(memoize=True)
def spacy_universal_embeddings(x):
  sent_encoder = spacy_universal_sentence_encoder.load_model('en_use_lg')
  x.simp_universal_doc = sent_encoder(x.simplified_snt)
  x.source_universal_doc = sent_encoder(x.source_snt)

  return x

# bjoern: few words per sentence~\cite{simpa}
def words_per_sentence(x, w_cnt, label):
    avg_cnt = len(x.simp_words)/len(x.simp_sentences)

    if label == SIMPLE:
      if avg_cnt <= w_cnt:
        return label
      else:
        return ABSTAIN
    else:
      if avg_cnt > w_cnt:
        return label
      else:
        return ABSTAIN
# bjoern: few words per sentence~\cite{simpa}
def make_word_cnt_lf(w_cnt, label=SIMPLE):

    return LabelingFunction(
        name=f"lf_words_cnt_wcount={w_cnt}_{label_map[label]}",
        f=words_per_sentence,
        resources=dict(w_cnt=w_cnt, label=label),
        pre=[spacy_nlp]
    )

#bjoern : high concreteness~\cite{simpa} avg
def avg_conreteness(x, con_threshold, label):

    con_list = []
    for c_token in x.simp_tokens:
      if c_token in concreteness_dic.keys():
        con_list.append(concreteness_dic[c_token])

    avg_con = np.mean(np.array(con_list))

    if label == SIMPLE:
      if avg_con >= con_threshold:
        return label
      else:
        return ABSTAIN
    else:
      if avg_con < con_threshold:
        return label
      else:
        return ABSTAIN

# bjoern: high concreteness~\cite{simpa}
def make_avg_conreteness_lf(con_threshold, label=SIMPLE):

    return LabelingFunction(
        name=f"lf_avg_concreteness={con_threshold}_{label_map[label]}",
        f=avg_conreteness,
        resources=dict(con_threshold=con_threshold, label=label),
        pre=[spacy_nlp]
    )

#bjoern : high concreteness~\cite{simpa} max
def max_conreteness(x, con_threshold, label):

    con_list = []
    for c_token in x.simp_tokens:
      if c_token in concreteness_dic.keys():
        con_list.append(concreteness_dic[c_token])

    max_con = np.max(np.array(con_list))

    if label == SIMPLE:
      if max_con >= con_threshold:
        return label
      else:
        return ABSTAIN
    else:
      if max_con < con_threshold:
        return label
      else:
        return ABSTAIN

# bjoern: high concreteness~\cite{simpa}
def make_max_conreteness_lf(con_threshold, label=SIMPLE):

    return LabelingFunction(
        name=f"lf_max_concreteness={con_threshold}_{label_map[label]}",
        f=max_conreteness,
        resources=dict(con_threshold=con_threshold, label=label),
        pre=[spacy_nlp]
    )

#bjoern : high concreteness~\cite{simpa} median
def median_conreteness(x, con_threshold, label):

    con_list = []
    for c_token in x.simp_tokens:
      if c_token in concreteness_dic.keys():
        con_list.append(concreteness_dic[c_token])

    median_con = np.median(np.array(con_list))

    if label == SIMPLE:
      if median_con >= con_threshold:
        return label
      else:
        return ABSTAIN
    else:
      if median_con < con_threshold:
        return label
      else:
        return ABSTAIN

# bjoern: high concreteness~\cite{simpa}
def make_median_conreteness_lf(con_threshold, label=SIMPLE):

    return LabelingFunction(
        name=f"lf_median_concreteness={con_threshold}_{label_map[label]}",
        f=median_conreteness,
        resources=dict(con_threshold=con_threshold, label=label),
        pre=[spacy_nlp]
    )

# bjoern : few content words (nouns, adjectives, verbs and adverbs) per sentence~\cite{simpa}
def content_words_ratio(x, ratio_threshold, label):

  tokens_data = x.simp_tokens_data
  content_cnt = 0
  for token_data in tokens_data:
    if token_data.pos_ in ['NOUN', 'ADJ', 'VERB', 'ADV']:
      content_cnt += 1
  content_ratio = content_cnt/len(tokens_data)

  if label == SIMPLE:
      if content_ratio <= ratio_threshold:
        return label
      else:
        return ABSTAIN
  else:
    if content_ratio > ratio_threshold:
      return label
    else:
      return ABSTAIN

# bjoern : few content words (nouns, adjectives, verbs and adverbs) per sentence~\cite{simpa}
def make_content_words_ratio_lf(ratio_threshold, label=SIMPLE):

    return LabelingFunction(
        name=f"lf_content_ratio_={ratio_threshold}_{label_map[label]}",
        f=content_words_ratio,
        resources=dict(ratio_threshold=ratio_threshold, label=label),
        pre=[spacy_nlp]
    )

# bjoern : few infrequent words~\cite{DBLP:conf/coling/StajnerH18}
def infrequent_words(x, infrequent_threshold, animal, label):
  animal_threshold = word_frequency(animal, 'en')
  infrequent_cnt = len([word for word in x.simp_tokens if word_frequency(word, 'en') < animal_threshold])


  if label == SIMPLE:
      if infrequent_cnt <= infrequent_threshold:
        return label
      else:
        return ABSTAIN
  else:
    if infrequent_cnt > infrequent_threshold:
      return label
    else:
      return ABSTAIN

def make_infrequent_words_lf(infrequent_threshold, animal, label=SIMPLE):

    return LabelingFunction(
        name=f"lf_infrequent_words_cnt={infrequent_threshold}_{animal}_{label_map[label]}",
        f=infrequent_words,
        resources=dict(infrequent_threshold=infrequent_threshold, animal=animal, label=label),
        pre=[spacy_nlp]
    )

#bjoern :low age of acquisition~\cite{simpa} avg
def avg_age_of_acquisition(x, age, label):

    aoas = []
    for c_token in x.simp_tokens:
      if c_token in aoa_dic.keys():
        aoas.append(aoa_dic[c_token])

    avg_aoa = np.mean(np.array(aoas))

    if label == SIMPLE:
      if avg_aoa <= age:
        return label
      else:
        return ABSTAIN
    else:
      if avg_aoa > age:
        return label
      else:
        return ABSTAIN
# bjoern: low age of acquisition~\cite{simpa}
def make_avg_age_of_acquisition_lf(age, label=SIMPLE):

    return LabelingFunction(
        name=f"lf_avg_age_of_acquisition={age}_{label_map[label]}",
        f=avg_age_of_acquisition,
        resources=dict(age=age, label=label),
        pre=[spacy_nlp]
    )

#low age of acquisition~\cite{simpa} max
def max_age_of_acquisition(x, age, label):

    aoas = []
    for c_token in x.simp_tokens:
      if c_token in aoa_dic.keys():
        aoas.append(aoa_dic[c_token])

    max_aoa = np.max(np.array(aoas))

    if label == SIMPLE:
      if max_aoa <= age:
        return label
      else:
        return ABSTAIN
    else:
      if max_aoa > age:
        return label
      else:
        return ABSTAIN
# bjoern: low age of acquisition~\cite{simpa}  max
def make_max_age_of_acquisition_lf(age, label=SIMPLE):

    return LabelingFunction(
        name=f"lf_max_age_of_acquisition={age}_{label_map[label]}",
        f=max_age_of_acquisition,
        resources=dict(age=age, label=label),
        pre=[spacy_nlp]
    )

#low age of acquisition~\cite{simpa} median
def median_age_of_acquisition(x, age, label):

    aoas = []
    for c_token in x.simp_tokens:
      if c_token in aoa_dic.keys():
        aoas.append(aoa_dic[c_token])

    median_aoa = np.median(np.array(aoas))

    if label == SIMPLE:
      if median_aoa <= age:
        return label
      else:
        return ABSTAIN
    else:
      if median_aoa > age:
        return label
      else:
        return ABSTAIN
# bjoern: low age of acquisition~\cite{simpa}
def make_median_age_of_acquisition_lf(age, label=SIMPLE):

    return LabelingFunction(
        name=f"lf_median_age_of_acquisition={age}_{label_map[label]}",
        f=median_age_of_acquisition,
        resources=dict(age=age, label=label),
        pre=[spacy_nlp]
    )

#bjoern :high imageability~\cite{simpa} avg
def avg_imageability(x, imageability_threshold, label):
    im_vals = []
    for c_token in x.simp_tokens:
      if c_token in imageability_dic.keys():
        im_vals.append(imageability_dic[c_token])

    avg_im = np.mean(np.array(im_vals))

    if label == SIMPLE:
      if avg_im >= imageability_threshold:
        return label
      else:
        return ABSTAIN
    else:
      if avg_im < imageability_threshold:
        return label
      else:
        return ABSTAIN

# bjoern: high imageability~\cite{simpa} avg
def make_avg_imageability_lf(imageability_threshold, label=SIMPLE):

    return LabelingFunction(
        name=f"lf_avg_imageability={imageability_threshold}_{label_map[label]}",
        f=avg_imageability,
        resources=dict(imageability_threshold=imageability_threshold, label=label),
        pre=[spacy_nlp]
    )

# bjoern: high imageability~\cite{simpa} median
def med_imageability(x, imageability_threshold, label):
    im_vals = []
    for c_token in x.simp_tokens:
      if c_token in imageability_dic.keys():
        im_vals.append(imageability_dic[c_token])

    med_im = np.median(np.array(im_vals))

    if label == SIMPLE:
      if med_im >= imageability_threshold:
        return label
      else:
        return ABSTAIN
    else:
      if med_im < imageability_threshold:
        return label
      else:
        return ABSTAIN

# bjoern: high imageability~\cite{simpa} median
def make_med_imageability_lf(imageability_threshold, label=SIMPLE):

    return LabelingFunction(
        name=f"lf_med_imageability={imageability_threshold}_{label_map[label]}",
        f=med_imageability,
        resources=dict(imageability_threshold=imageability_threshold, label=label),
        pre=[spacy_nlp]
    )

# frequency of nominalisations~\cite{textevaluator}, implemented with~\citet{klein2020qanom}
def freq_nominalisations(x, thresh, label):
  countElements = len(nom_detector(x.simplified_snt, threshold=0.75, return_probability=False))
  
  if label == SIMPLE:
      if countElements <= thresh:
        return label
      else:
        return ABSTAIN
  else:
    if countElements > thresh:
      return label
    else:
      return ABSTAIN

def make_freq_nominalisations_lf(thresh, label=SIMPLE):

    return LabelingFunction(
        name=f"freq_nominalisations_{label}_{thresh}",
        f=freq_nominalisations,
        resources=dict(thresh=thresh, label=label),
        pre=[spacy_nlp]
    ) 

# Fabian : high percentage of vocabulary learned in initial stages of foreign language learning~\cite{tanaka} $\rightarrow$ language proficiency test
def perc_vocab_initial_forLang_learn(x, thresh, label):
  ratio = len([w for w in x.simp_doc if w.text.lower() in ox5k_a])/len(x.simp_tokens)
  if label == SIMPLE:
      if ratio <= thresh:
        return label
      else:
        return ABSTAIN
  else:
    if ratio > thresh:
      return label
    else:
      return ABSTAIN

def make_perc_vocab_initial_forLang_learn_lf(thresh, label=SIMPLE):

    return LabelingFunction(
        name=f"perc_vocab_initial_forLang_learn_{label}_{thresh}",
        f=perc_vocab_initial_forLang_learn,
        resources=dict(thresh=thresh, label=label),
        pre=[spacy_nlp]
    )

# Fabian: word length - frequency per thousand words of words containing more than eight characters~\cite{textevaluator}

def perc_more_than_8_characters(x, thresh, label):
  freqElements = len([w for w in x.simp_doc if len(w)>8])/len(x.simp_doc)

  if label == SIMPLE:
      if freqElements <= thresh:
        return label
      else:
        return ABSTAIN
  else:
    if freqElements > thresh:
      return label
    else:
      return ABSTAIN
  
def make_perc_more_than_8_characters_lf(thresh, label=SIMPLE):

    return LabelingFunction(
        name=f"perc_more_than_8_characters_{label}_{thresh}",
        f=perc_more_than_8_characters,
        resources=dict(thresh=thresh, label=label),
        pre=[spacy_nlp]
    )


# Fabian: Frequency of negation 
def freq_negations(x, thresh, label):
  countElements = len([tok for tok in x.simp_doc if tok.dep_ == 'neg'])

  if label == SIMPLE:
      if countElements <= thresh:
        return label
      else:
        return ABSTAIN
  else:
    if countElements > thresh:
      return label
    else:
      return ABSTAIN
  
def make_freq_negations_lf(thresh, label=SIMPLE):

    return LabelingFunction(
        name=f"freq_negations_{label}_{thresh}",
        f=freq_negations,
        resources=dict(thresh=thresh, label=label),
        pre=[spacy_nlp]
    )



# Fabian: frequency of third person singular pronouns~\cite{textevaluator}
def freq_third_person_singular_pronouns(x, thresh, label):
  countElements = 0
  for token in x.simp_doc:
    if  token.pos_ == "PRON" and token.morph.get("Person") == ["3"] and token.morph.get("Number") == ['Sing']:
        
        countElements+=1

  if label == SIMPLE:
      if countElements <= thresh:
        return label
      else:
        return ABSTAIN
  else:
    if countElements > thresh:
      return label
    else:
      return ABSTAIN
  
def make_freq_third_person_singular_pronouns_lf(thresh, label=SIMPLE):

    return LabelingFunction(
        name=f"freq_third_person_singular_pronouns_{label}_{thresh}",
        f=freq_third_person_singular_pronouns,
        resources=dict(thresh=thresh, label=label),
        pre=[spacy_nlp]
    )


# Fabian: frequency of past tense aspect verbs~\cite{textevaluator}
def num_past_tense(x, thresh, label):
  num_w = len([w for w in x.simp_doc if w.tag_ == "VBD"])
  if label == SIMPLE:
      if num_w <= thresh:
        return label
      else:
        return ABSTAIN
  else:
    if num_w > thresh:
      return label
    else:
      return ABSTAIN

def make_num_past_tense_lf(thresh, label=SIMPLE):

    return LabelingFunction(
        name=f"num_past_tense_{label}_{thresh}",
        f=num_past_tense,
        resources=dict(thresh=thresh, label=label),
        pre=[spacy_nlp]
    )

# Fabian: percentage of past tense aspect verbs~\cite{textevaluator}
def perc_past_tense(x, thresh, label):
  num_w = len([w for w in x.simp_doc if w.tag_ == "VBD"])/max(1,len([w for w in x.simp_doc if w.pos_ == "VERB"]))
  if label == SIMPLE:
      if num_w <= thresh:
        return label
      else:
        return ABSTAIN
  else:
    if num_w > thresh:
      return label
    else:
      return ABSTAIN

def make_perc_past_tense_lf(thresh, label=SIMPLE):

    return LabelingFunction(
        name=f"perc_past_tense_{label}_{thresh}",
        f=perc_past_tense,
        resources=dict(thresh=thresh, label=label),
        pre=[spacy_nlp]
    )


# Fabian: frequency of past perfect verbs~\cite{textevaluator}
def num_past_perfect(x, thresh, label):
  num_w = len([w for w in x.simp_doc if w.tag_ == "VBN"])
  if label == SIMPLE:
      if num_w <= thresh:
        return label
      else:
        return ABSTAIN
  else:
    if num_w > thresh:
      return label
    else:
      return ABSTAIN

def make_num_past_perfect_lf(thresh, label=SIMPLE):

    return LabelingFunction(
        name=f"num_past_perfect_{label}_{thresh}",
        f=num_past_perfect,
        resources=dict(thresh=thresh, label=label),
        pre=[spacy_nlp]
    )


# Fabian: percentage of past perfect aspect verbs~\cite{textevaluator}
def perc_past_perfect(x, thresh, label):
  num_w = len([w for w in x.simp_doc if w.tag_ == "VBN"])/ max(1,len([w for w in x.simp_doc if w.pos_ == "VERB"]))
  if label == SIMPLE:
      if num_w <= thresh:
        return label
      else:
        return ABSTAIN
  else:
    if num_w > thresh:
      return label
    else:
      return ABSTAIN

def make_perc_past_perfect_lf(thresh, label=SIMPLE):

    return LabelingFunction(
        name=f"perc_past_perfect_{label}_{thresh}",
        f=perc_past_perfect,
        resources=dict(thresh=thresh, label=label),
        pre=[spacy_nlp]
    )


# Fabian: average number of words before the main verb~\cite{textevaluator}
def avg_num_words_before_main_verb(x, thresh, label):
  for sentence in x.simp_doc.sents:
    if not "ROOT" in [token.dep_ for token in sentence]:
      return ABSTAIN
  num_w = np.mean([[token.dep_ for token in sentence].index("ROOT") for sentence in x.simp_doc.sents])
  if label == SIMPLE:
      if num_w <= thresh:
        return label
      else:
        return ABSTAIN
  else:
    if num_w > thresh:
      return label
    else:
      return ABSTAIN

def make_avg_num_words_before_main_verb_lf(thresh, label=SIMPLE):

    return LabelingFunction(
        name=f"avg_num_words_before_main_verb_{label}_{thresh}",
        f=avg_num_words_before_main_verb,
        resources=dict(thresh=thresh, label=label),
        pre=[spacy_nlp]
    )


# Fabian: low entity to token ratio per text\cite{DBLP:conf/dsai/StajnerNI20}
def entity_token_ratio_text(x, thresh, label):
  ratio = len(x.simp_entities)/len(x.simp_tokens)
  if label == SIMPLE:
      if ratio <= thresh:
        return label
      else:
        return ABSTAIN
  else:
    if ratio > thresh:
      return label
    else:
      return ABSTAIN

def make_entity_token_ratio_text_lf(thresh, label=SIMPLE):

    return LabelingFunction(
        name=f"entity_token_ratio_text_{label}_{thresh}",
        f=entity_token_ratio_text,
        resources=dict(thresh=thresh, label=label),
        pre=[spacy_nlp]
    )

# Fabian: low entity to token ratio per sentence\cite{DBLP:conf/dsai/StajnerNI20}
def entity_token_ratio_sentence(x, thresh, label):
  if len(x.simp_sentences)<=1:
    ratio = len(x.simp_entities)/len(x.simp_tokens)
    if label == SIMPLE:
        if ratio <= thresh:
          return label
        else:
          return ABSTAIN
    else:
      if ratio > thresh:
        return label
      else:
        return ABSTAIN
  else:
    ratios = []
    for sentence in x.simp_doc.sents:
      s_tokens = [token for token in sentence]
      num_ents =[token.ent_iob_ for token in sentence].count("B")
      ratio = num_ents/len(s_tokens)
      ratios.append(ratio)
    avg_ratios = sum(ratios)/len(ratios)


    if label == SIMPLE:
        if avg_ratios <= thresh:
          return label
        else:
          return ABSTAIN
    else:
      if avg_ratios > thresh:
        return label
      else:
        return ABSTAIN

def make_entity_token_ratio_sentence_lf(thresh, label=SIMPLE):

    return LabelingFunction(
        name=f"entity_token_ratio_sentence_{label}_{thresh}",
        f=entity_token_ratio_sentence,
        resources=dict(thresh=thresh, label=label),
        pre=[spacy_nlp]
    )

# Fabian: low entity to token ratio per paragraph\cite{DBLP:conf/dsai/StajnerNI20}
def entity_token_ratio_paragraph(x, thresh, label):
  if len(x.simp_paragraph_tokens_data)<=1:
    ratio = len(x.simp_entities)/len(x.simp_tokens)
    if label == SIMPLE:
        if ratio <= thresh:
          return label
        else:
          return ABSTAIN
    else:
      if ratio > thresh:
        return label
      else:
        return ABSTAIN
  else:
    ratios = []
    for paragraph in x.simp_paragraph_tokens_data:
      s_tokens = [token for token in paragraph]
      num_ents = [token.ent_iob_ for token in paragraph].count("B")
      if len(s_tokens) == 0:
        return ABSTAIN
      ratio = num_ents/len(s_tokens)
      ratios.append(ratio)
    avg_ratios = sum(ratios)/len(ratios)

    if label == SIMPLE:
        if avg_ratios <= thresh:
          return label
        else:
          return ABSTAIN
    else:
      if avg_ratios > thresh:
        return label
      else:
        return ABSTAIN

def make_entity_token_ratio_paragraph_lf(thresh, label=SIMPLE):

    return LabelingFunction(
        name=f"entity_token_ratio_paragraph_{label}_{thresh}",
        f=entity_token_ratio_paragraph,
        resources=dict(thresh=thresh, label=label),
        pre=[spacy_nlp_paragraph]
    )

# Fabian: frequency per thousand words/ratio of all words on Academic Word List \url{https://www.eapfoundation.com/vocab/academic/awllists/}~\cite{textevaluator}
def ratio_academic_word_list(x, thresh, label):
  ratio_awl = len([w for w in x.simp_words if w.lower() in academic_word_list])/len(x.simp_words)
  if label == SIMPLE:
      if ratio_awl <= thresh:
        return label
      else:
        return ABSTAIN
  else:
    if ratio_awl > thresh:
      return label
    else:
      return ABSTAIN

def make_ratio_academic_word_list_lf(thresh, label=SIMPLE):

    return LabelingFunction(
        name=f"ratio_academic_word_list_{label}_{thresh}",
        f=ratio_academic_word_list,
        resources=dict(thresh=thresh, label=label),
        pre=[spacy_nlp]
    )


# Fabian: Average lexical richness~\cite{DBLP:conf/lrec/StajnerNH20}
# as: average number of unique lemmas per sentence
def num_unique_lemmas(x, thresh, label):
  avg_lemmas = np.mean([len(set([w.lemma_ for w in sent])) for sent in x.simp_doc.sents])
  if label == SIMPLE:
      if avg_lemmas <= thresh:
        return label
      else:
        return ABSTAIN
  else:
    if avg_lemmas > thresh:
      return label
    else:
      return ABSTAIN

def make_num_unique_lemmas_lf(thresh, label=SIMPLE):

    return LabelingFunction(
        name=f"num_unique_lemmas_{label}_{thresh}",
        f=num_unique_lemmas,
        resources=dict(thresh=thresh, label=label),
        pre=[spacy_nlp]
    )

# Fabian: Average lexical richness~\cite{DBLP:conf/lrec/StajnerNH20}
# as: average number of unique lemmas per sentence per number of tokens per sentence
def num_unique_lemmas_norm(x, thresh, label):
  avg_lemmas = np.mean([len(set([w.lemma_ for w in sent]))/len(sent) for sent in x.simp_doc.sents])
  if label == SIMPLE:
      if avg_lemmas <= thresh:
        return label
      else:
        return ABSTAIN
  else:
    if avg_lemmas > thresh:
      return label
    else:
      return ABSTAIN

def make_num_unique_lemmas_norm_lf(thresh, label=SIMPLE):

    return LabelingFunction(
        name=f"num_unique_lemmas_norm_{label}_{thresh}",
        f=num_unique_lemmas_norm,
        resources=dict(thresh=thresh, label=label),
        pre=[spacy_nlp]
    )

# Fabian : low depth of the syntactic tree~\cite{DBLP:conf/lrec/StajnerNH20}
def depth_of_syntactic_tree(x, thresh, label):
  doc = x.simp_doc
  depths = {}

  def walk_tree(node, depth):
      depths[node.orth_] = depth
      if node.n_lefts + node.n_rights > 0:
          return [walk_tree(child, depth + 1) for child in node.children]
  
  [walk_tree(sent.root, 0) for sent in doc.sents]
  max_depth = max(depths.values())

  if label == SIMPLE:
      if max_depth <= thresh:
        return label
      else:
        return ABSTAIN
  else:
    if max_depth > thresh:
      return label
    else:
      return ABSTAIN

def make_depth_of_syntactic_tree_lf(thresh, label=SIMPLE):

    return LabelingFunction(
        name=f"depth_of_syntactic_tree_{label}_{thresh}",
        f=depth_of_syntactic_tree,
        resources=dict(thresh=thresh, label=label),
        pre=[spacy_nlp]
    )

# Fabian : low number of punctuation in text~\cite{DBLP:conf/dsai/StajnerNI20}
def avg_num_punctuation_text(x, thresh, label):
  avg_num_punc = np.mean([[tok.pos_ for tok in sent].count("PUNCT") for sent in x.simp_doc.sents])
  if label == SIMPLE:
      if avg_num_punc <= thresh:
        return label
      else:
        return ABSTAIN
  else:
    if avg_num_punc > thresh:
      return label
    else:
      return ABSTAIN

def make_avg_num_punctuation_text_lf(thresh, label=SIMPLE):

    return LabelingFunction(
        name=f"avg_num_punctuation_text_{label}_{thresh}",
        f=avg_num_punctuation_text,
        resources=dict(thresh=thresh, label=label),
        pre=[spacy_nlp]
    )

# Fabian : low number of unique entities in text~\cite{DBLP:conf/dsai/StajnerNI20}
def unique_entities_text(x, thresh, label):
  num_unique_ents = len(set(x.simp_entities))
  if label == SIMPLE:
      if num_unique_ents <= thresh:
        return label
      else:
        return ABSTAIN
  else:
    if num_unique_ents > thresh:
      return label
    else:
      return ABSTAIN

def make_unique_entities_text_lf(thresh, label=SIMPLE):

    return LabelingFunction(
        name=f"unique_entities_text_{label}_{thresh}",
        f=unique_entities_text,
        resources=dict(thresh=thresh, label=label),
        pre=[spacy_nlp]
    )

# Fabian : low average number of unique entities per sentence~\cite{DBLP:conf/dsai/StajnerNI20}
def average_entities_sentence(x, thresh, label):
  avgs = []
  for sentence in x.simp_doc.sents:
      entities = entities_in_list_of_tokens(sentence)
      avgs.append(len(set(entities)))
  if len(avgs) > 0:
    num_unique_ents = sum(avgs)/len(avgs)
  else:
    num_unique_ents = 0

  if label == SIMPLE:
      if num_unique_ents <= thresh:
        return label
      else:
        return ABSTAIN
  else:
    if num_unique_ents > thresh:
      return label
    else:
      return ABSTAIN

def make_average_entities_sentence_lf(thresh, label=SIMPLE):

    return LabelingFunction(
        name=f"average_entities_sentence_{label}_{thresh}",
        f=average_entities_sentence,
        resources=dict(thresh=thresh, label=label),
        pre=[spacy_nlp]
    )

# Fabian : low average number of unique entities per paragraph~\cite{DBLP:conf/dsai/StajnerNI20}
def average_entities_paragraph(x, thresh, label):
  avgs = []
  for paragraph in x.simp_paragraph_tokens_data:
      entities = entities_in_list_of_tokens(paragraph)
      avgs.append(len(set(entities)))

  if len(avgs) > 0:
    num_unique_ents = sum(avgs)/len(avgs)
  else:
    num_unique_ents = 0

  if label == SIMPLE:
      if num_unique_ents <= thresh:
        return label
      else:
        return ABSTAIN
  else:
    if num_unique_ents > thresh:
      return label
    else:
      return ABSTAIN

def make_average_entities_paragraph_lf(thresh, label=SIMPLE):

    return LabelingFunction(
        name=f"average_entities_paragraph_{label}_{thresh}",
        f=average_entities_paragraph,
        resources=dict(thresh=thresh, label=label),
        pre=[spacy_nlp_paragraph]
    )

#low unique entities to total number of entities ratio per text/sentence (avg) /paragraph (avg)~\cite{DBLP:conf/dsai/StajnerNI20}
# Fabian: low unique entity to total num entities ratio per text\cite{DBLP:conf/dsai/StajnerNI20}
def unique_entity_total_entity_ratio_text(x, thresh, label):
  if len(x.simp_entities)>0:
    ratio = len(set(x.simp_entities))/len(x.simp_entities)
  else:
    ratio = 1

  if label == SIMPLE:
      if ratio <= thresh:
        return label
      else:
        return ABSTAIN
  else:
    if ratio > thresh:
      return label
    else:
      return ABSTAIN

def make_unique_entity_total_entity_ratio_text_lf(thresh, label=SIMPLE):

    return LabelingFunction(
        name=f"unique_entity_total_entity_ratio_text_{label}_{thresh}",
        f=unique_entity_total_entity_ratio_text,
        resources=dict(thresh=thresh, label=label),
        pre=[spacy_nlp]
    )

# Fabian: low unique entity to total num entities ratio per sentence\cite{DBLP:conf/dsai/StajnerNI20}
def unique_entity_total_entity_ratio_sentence(x, thresh, label):
  if len(x.simp_entities)>0 and len(x.simp_sentences)<=1:
    ratio = len(set(x.simp_entities))/len(x.simp_entities)
    if label == SIMPLE:
        if ratio <= thresh:
          return label
        else:
          return ABSTAIN
    else:
      if ratio > thresh:
        return label
      else:
        return ABSTAIN
  else:
    ratios = []
    for sentence in x.simp_doc.sents:
      entities = entities_in_list_of_tokens(sentence)
      if len(entities) == 0:
        ratios.append(1)
      else:
        ratios.append(len(set(entities))/len(entities))

    if len(ratios) > 0:
      avg_ratios = sum(ratios)/len(ratios)
    else:
      avg_ratios = 1

    if label == SIMPLE:
        if avg_ratios <= thresh:
          return label
        else:
          return ABSTAIN
    else:
      if avg_ratios > thresh:
        return label
      else:
        return ABSTAIN

def make_unique_entity_total_entity_ratio_sentence_lf(thresh, label=SIMPLE):

    return LabelingFunction(
        name=f"unique_entity_total_entity_ratio_sentence{thresh}label={label}",
        f=unique_entity_total_entity_ratio_sentence,
        resources=dict(thresh=thresh, label=label),
        pre=[spacy_nlp]
    )

# Fabian: low unique entity to total num entities ratio per paragraph\cite{DBLP:conf/dsai/StajnerNI20}
def unique_entity_total_entity_ratio_paragraph(x, thresh, label):
  if len(x.simp_paragraph_tokens_data)<=1:
    if len(x.simp_entities)>0:
      ratio = len(set(x.simp_entities))/len(x.simp_entities)
    else:
      ratio = 1

    if label == SIMPLE:
        if ratio <= thresh:
          return label
        else:
          return ABSTAIN
    else:
      if ratio > thresh:
        return label
      else:
        return ABSTAIN
  else:
    ratios = []
    for paragraph in x.simp_paragraph_tokens_data:
      entities = entities_in_list_of_tokens(paragraph)
      if len(entities) == 0:
        ratios.append(1)
      else:
        ratios.append(len(set(entities))/len(entities))

    if len(ratios)>0:
      avg_ratios = sum(ratios)/len(ratios)
    else:
      avg_ratios = 1

    if label == SIMPLE:
        if avg_ratios <= thresh:
          return label
        else:
          return ABSTAIN
    else:
      if avg_ratios > thresh:
        return label
      else:
        return ABSTAIN

def make_unique_entity_total_entity_ratio_paragraph_lf(thresh, label=SIMPLE):

    return LabelingFunction(
        name=f"unique_entity_total_entity_ratio_paragraph_{label}_{thresh}",
        f=unique_entity_total_entity_ratio_paragraph,
        resources=dict(thresh=thresh, label=label),
        pre=[spacy_nlp_paragraph]
    )

# Fabian: few relative-clauses for people with poor language skills~\cite{arfe}
def no_relative_clauses(x, thresh, label):
  rel_pron = ["which", "that", "whose", "whoever", "whomever", "who", "whom"]
  all_tokens = [(a.text, a.text.lower() in rel_pron, a.pos_) for a in x.simp_tokens_data]
  all_pron_tokens = [(a.text, a.text.lower() in rel_pron, a.pos_) for a in x.simp_tokens_data if a.pos_ == "PRON"]
  num_rel_pronouns = len([a for a in all_pron_tokens if a[1]])

  if label == SIMPLE:
      if num_rel_pronouns <= thresh:
        return label
      else:
        return ABSTAIN
  else:
    if num_rel_pronouns > thresh:
      return label
    else:
      return ABSTAIN

def make_no_relative_clauses_lf(thresh, label=SIMPLE):

    return LabelingFunction(
        name=f"no_relative_clauses_{label}_{thresh}",
        f=no_relative_clauses,
        resources=dict(thresh=thresh, label=label),
        pre=[spacy_nlp_paragraph]
    )

# Fabian: few relative-sub-clauses for people with poor language skills~\cite{arfe}
def no_relative_sub_clauses(x, thresh, label):
  rel_pron = ["which", "that", "whose", "whoever", "whomever", "who", "whom"]

  true_relative_sub_clauses = []
  for i, tok in enumerate(x.simp_tokens_data):
    if tok.pos_ == "PRON" and tok.text.lower() in rel_pron:
      if i>0 and not x.simp_tokens_data[i-1].is_sent_end:
        true_relative_sub_clauses.append(tok)

  num_rel_pronouns = len(true_relative_sub_clauses)


  if label == SIMPLE:
      if num_rel_pronouns <= thresh:
        return label
      else:
        return ABSTAIN
  else:
    if num_rel_pronouns > thresh:
      return label
    else:
      return ABSTAIN

def make_no_relative_sub_clauses_lf(thresh, label=SIMPLE):

    return LabelingFunction(
        name=f"no_relative_sub_clauses_{label}_{thresh}",
        f=no_relative_sub_clauses,
        resources=dict(thresh=thresh, label=label),
        pre=[spacy_nlp_paragraph]
    )

# Fabian: no anaphors for people with language problems~\cite{arfe}
def few_anaphors(x, thresh, label):

  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    prediction = predictor.predict(document=x.simp_doc.text)  # get prediction
    a_counter = 0
    for c in prediction['clusters']:
      c_test = True
      for i in c:
        span_words = [a.text for a in x.simp_doc[i[0]:i[1]+1]]
        span_pos = [a.pos_ for a in x.simp_doc[i[0]:i[1]+1]]
        pos_test = len([b for b in span_pos if b in ["NOUN", "PRON", "PROPN"]])
        if pos_test == 0:
          c_test = False
      if c_test:
        a_counter += 1


    if label == SIMPLE:
        if a_counter <= thresh:
          return label
        else:
          return ABSTAIN
    else:
      if a_counter > thresh:
        return label
      else:
        return ABSTAIN

def make_few_anaphors_lf(thresh, label=SIMPLE):

    return LabelingFunction(
        name=f"few_anaphors_{label}_{thresh}",
        f=few_anaphors,
        resources=dict(thresh=thresh, label=label),
        pre=[spacy_nlp_paragraph]
    )

# Fabian: low number of cases with max distance paragraph between 2 appearances of same entity~\cite{DBLP:conf/dsai/StajnerNI20}
def distance_appearance_same_entities_paragraph(x, thresh_distance, thresh_number, label=SIMPLE):
  count_num_appearances_max_or_higher = 0

  list_of_par_entities = []
  for paragraph in x.simp_paragraph_tokens_data:
    curr_par_ents = []
    for i, token in enumerate(paragraph):
      curr_ent = ""
      if token.ent_iob_ == "B":
        curr_ent += token.text
        for tok2 in paragraph[i+1:len(paragraph)]:
          if tok2.ent_iob_ == "I":
            curr_ent += " "+tok2.text
          else:
            break
        curr_par_ents.append(curr_ent)
    list_of_par_entities.append(curr_par_ents)

  ent_count = 0
  for i, liste in enumerate(list_of_par_entities):
    for ent in liste:
      ent_max_distance = 0
      for e, liste2 in enumerate(list_of_par_entities[i+1:len(list_of_par_entities)]):
        if ent in liste2:
          ent_max_distance = e-i
          if ent_max_distance >= thresh_distance:
            ent_count += 1
          break

  if label == SIMPLE:
      if ent_count < thresh_number:
        return label
      else:
        return ABSTAIN
  else:
    if ent_count >= thresh_number:
      return label
    else:
      return ABSTAIN

def make_distance_appearance_same_entities_paragraph_lf(thresh_distance, thresh_number, label=SIMPLE):

    return LabelingFunction(
        name=f"distance_appearance_same_entities_paragraph_dist_{label}_{thresh_distance}_num{thresh_number}",
        f=distance_appearance_same_entities_paragraph,
        resources=dict(thresh_distance= thresh_distance, thresh_number=thresh_number,  label=label),
        pre=[spacy_nlp_paragraph]
    )

# Fabian: low avg distance paragraphs between all pairs of same entities~\cite{DBLP:conf/dsai/StajnerNI20}
def avarage_distance_appearance_same_entities_paragraph(x, thresh, label=SIMPLE):

  list_of_par_entities = []
  for paragraph in x.simp_paragraph_tokens_data:
    curr_par_ents = []
    for i, token in enumerate(paragraph):
      curr_ent = ""
      if token.ent_iob_ == "B":
        curr_ent += token.text
        for tok2 in paragraph[i+1:len(paragraph)]:
          if tok2.ent_iob_ == "I":
            curr_ent += " "+tok2.text
          else:
            break
        curr_par_ents.append(curr_ent)
    list_of_par_entities.append(curr_par_ents)

  res_dict = {}
  for i, liste in enumerate(list_of_par_entities):
    for ent in liste:
      if not ent in res_dict.keys():
        res_dict[ent] = []
      l_for_avg = []
      for e, liste2 in enumerate(list_of_par_entities[i+1:len(list_of_par_entities)]):
        if ent in liste2:
          res_dict[ent].append(e-i)

    res_avg_dict = {}
    if len(res_dict) > 0:
      for ent in res_dict.keys():
          if len(res_dict[ent])>0:
             res_avg_dict[ent] = sum(res_dict[ent])/len(res_dict[ent])

  if len(res_avg_dict.values())>0:
    avg_dist = sum(res_avg_dict.values())/len(res_avg_dict.values())
  else:
    avg_dist = 0


  if label == SIMPLE:
      if avg_dist <= thresh:
        return label
      else:
        return ABSTAIN
  else:
    if avg_dist > thresh:
      return label
    else:
      return ABSTAIN

def make_average_distance_appearance_same_entities_paragraph_lf(thresh, label=SIMPLE):

    return LabelingFunction(
        name=f"avarage_distance_appearance_same_entities_paragraph_{label}_{thresh}",
        f=avarage_distance_appearance_same_entities_paragraph,
        resources=dict(thresh=thresh,  label=label),
        pre=[spacy_nlp_paragraph]
    )

# Fabian: low avg distance sentence between all pairs of same entities~\cite{DBLP:conf/dsai/StajnerNI20}
def avarage_distance_appearance_same_entities_sentence(x, thresh, label=SIMPLE):

  list_of_par_entities = []
  for paragraph in x.simp_doc.sents:
    curr_par_ents = []
    for i, token in enumerate(paragraph):
      curr_ent = ""
      if token.ent_iob_ == "B":
        curr_ent += token.text
        for tok2 in paragraph[i+1:len(paragraph)]:
          if tok2.ent_iob_ == "I":
            curr_ent += " "+tok2.text
          else:
            break
        curr_par_ents.append(curr_ent)
    list_of_par_entities.append(curr_par_ents)

  res_dict = {}
  for i, liste in enumerate(list_of_par_entities):
    for ent in liste:
      if not ent in res_dict.keys():
        res_dict[ent] = []
      l_for_avg = []
      for e, liste2 in enumerate(list_of_par_entities[i+1:len(list_of_par_entities)]):
        if ent in liste2:
          res_dict[ent].append(e-i)

    res_avg_dict = {}
    if len(res_dict) > 0:
      for ent in res_dict.keys():
          if len(res_dict[ent])>0:
             res_avg_dict[ent] = sum(res_dict[ent])/len(res_dict[ent])

  if len(res_avg_dict.values())>0:
    avg_dist = sum(res_avg_dict.values())/len(res_avg_dict.values())
  else:
    avg_dist = 0


  if label == SIMPLE:
      if avg_dist <= thresh:
        return label
      else:
        return ABSTAIN
  else:
    if avg_dist > thresh:
      return label
    else:
      return ABSTAIN

def make_average_distance_appearance_same_entities_sentence_lf(thresh, label=SIMPLE):

    return LabelingFunction(
        name=f"avarage_distance_appearance_same_entities_sentence_{label}_{thresh}",
        f=avarage_distance_appearance_same_entities_sentence,
        resources=dict(thresh=thresh,  label=label),
        pre=[spacy_nlp]
    )

# Fabian: low number of cases with max distance sentence between 2 appearances of same entity~\cite{DBLP:conf/dsai/StajnerNI20}
def distance_appearance_same_entities_sentence(x, thresh_distance, thresh_number, label=SIMPLE):
  count_num_appearances_max_or_higher = 0

  list_of_par_entities = []
  for paragraph in x.simp_doc.sents:
    curr_par_ents = []
    for i, token in enumerate(paragraph):
      curr_ent = ""
      if token.ent_iob_ == "B":
        curr_ent += token.text
        for tok2 in paragraph[i+1:len(paragraph)]:
          if tok2.ent_iob_ == "I":
            curr_ent += " "+tok2.text
          else:
            break
        curr_par_ents.append(curr_ent)
    list_of_par_entities.append(curr_par_ents)

  ent_count = 0
  for i, liste in enumerate(list_of_par_entities):
    for ent in liste:
      ent_max_distance = 0
      for e, liste2 in enumerate(list_of_par_entities[i+1:len(list_of_par_entities)]):
        if ent in liste2:
          ent_max_distance = e-i
          if ent_max_distance >= thresh_distance:
            ent_count += 1
          break

  if label == SIMPLE:
      if ent_count < thresh_number:
        return label
      else:
        return ABSTAIN
  else:
    if ent_count >= thresh_number:
      return label
    else:
      return ABSTAIN

def make_distance_appearance_same_entities_sentence_lf(thresh_distance, thresh_number, label=SIMPLE):

    return LabelingFunction(
        name=f"distance_appearance_same_entities_sentence_dist_{label}_{thresh_distance}_num{thresh_number}",
        f=distance_appearance_same_entities_sentence,
        resources=dict(thresh_distance= thresh_distance, thresh_number=thresh_number,  label=label),
        pre=[spacy_nlp_paragraph]
    )

# Fabian: high average distance (in sentences, paragraphs) between consecutive entities ~\cite{DBLP:conf/dsai/StajnerNI20}

def avarage_distance_entities(x, thresh, scope, same_or_consecutive, label=SIMPLE):

  return ABSTAIN

  #local variable 'e' referenced before assignment

  thresh = 1
  # scope "sent" or else
  sel = same_or_consecutive # consec or else


  if scope == "sent":
    choice = x.simp_doc.sents
  else:
    choice = x.simp_paragraph_tokens_data

  ent_list = []
  for paragraph in choice:
    curr_par_ents = []
    curr_par_ents_positions = []
    for i, token in enumerate(paragraph):
      curr_ent = ""
      if token.ent_iob_ == "B":
        curr_ent += token.text
        for e, tok2 in enumerate(paragraph[i+1:len(paragraph)]):
          if tok2.ent_iob_ == "I":
            curr_ent += " "+tok2.text
          else:
            break
        #local variable 'e' referenced before assignment
        curr_par_ents_positions.append({"text":curr_ent, "beg": i, "end": i+e})
    ent_list.append(curr_par_ents_positions)


  if sel == "consec":
    total_distances = []
    for sent in ent_list:
      dis_list = []
      for i, e in enumerate(sent):
        if len(sent)>i+1:
          dis_list.append(sent[i+1]["end"]-e["end"])
      if len(dis_list)>1:
        #print(">1",dis_list)
        total_distances.append((sum(dis_list))/(len(dis_list)))
      elif len(dis_list) == 1:
        #print("=1",dis_list)
        total_distances.append(sum(dis_list))
      else:
        #print("=0",dis_list)
        total_distances.append(0)
    check = sum(total_distances)/len(total_distances)

    if check == 0:
        return ABSTAIN

    if label == SIMPLE:
      if check < thresh:
        return label
      else:
        return ABSTAIN
    else:
      if check >= thresh:
        return label
      else:
        return ABSTAIN

  else:
    total_list = []
    for sent in ent_list: #alle sätze
      dis_list = []
      done = []
      for i, e in enumerate(sent): # alle antitäten angucken aber
        if e["text"] not in done: # jede ent nur einmal prosatz
          if len(sent)>i+1: #wenn es noch einen nächsten GIBT
            for i2, e2 in enumerate(sent[i+1:]): # gucke auf alle andere ents
              if e["text"].lower() == e2["text"].lower():
                dis_list.append(sent[i+i2+1]["end"]-e["end"])
          done.append(e["text"])
      if len(dis_list)< 1:
        total_list.append(0)
      else:
        total_list.append(sum(dis_list)/len(dis_list))

    check = sum(total_list)/len(total_list)

    if check == 0:
        return ABSTAIN

    if label == SIMPLE:
      if check < thresh:
        return label
      else:
        return ABSTAIN
    else:
      if check >= thresh:
        return label
      else:
        return ABSTAIN

def make_avarage_distance_entities_lf(thresh, scope, same_or_consecutive, label=SIMPLE):

    return LabelingFunction(
        name=f"avarage_distance_entities_{scope}_{same_or_consecutive}_{thresh}_{label}",
        f=avarage_distance_entities,
        resources=dict(thresh= thresh, scope=scope, same_or_consecutive=same_or_consecutive,  label=label),
        pre=[spacy_nlp_paragraph]
    )

# christin: low proportion of long (letters, syllables) words~\cite{arfe}
def proportion_of_long_words_syllables(x, proportion, long_length, label):
  number_of_words = len(x.simp_words)
  number_long_words = 0
  for ct in x.simp_syllables_cnt:
    if ct is not None:
      if ct >= long_length:
        number_long_words += 1
  actual_proportion = number_long_words/number_of_words


  if label == SIMPLE:
    if actual_proportion <= proportion:
      return label
    else:
      return ABSTAIN
  else:
    if actual_proportion > proportion:
      return label
    else:
      return ABSTAIN

def low_proportion_of_long_words_syllables(long_length, proportion, label):
  return LabelingFunction(
      name=f"low_prop_long_words_syllables_long={long_length}_prop={proportion}",
      f=proportion_of_long_words_syllables,
      resources=dict(proportion=proportion, long_length=long_length,
                     label=label), pre=[spacy_nlp]
  )


# christin: low proportion of long (letters, syllables) words~\cite{arfe}
def proportion_of_long_words_letters(x, proportion, long_length, label):
  number_of_words = len(x.simp_words)
  number_long_words = 0
  for w in x.simp_words:
    if len(w) >= long_length:
      number_long_words += 1
  actual_proportion = number_long_words/number_of_words

  if label == SIMPLE:
    if actual_proportion <= proportion:
      return label
    else:
      return ABSTAIN
  else:
    if actual_proportion > proportion:
      return label
    else:
      return ABSTAIN

def low_proportion_of_long_words_letters(long_length, proportion, label):
  return LabelingFunction(
      name=f"low_prop_long_words_letters_long={long_length}_prop={proportion}",
      f=proportion_of_long_words_letters,
      resources=dict(proportion=proportion, long_length=long_length,
                     label=label), pre=[spacy_nlp]
  )

# christin: low Flesch-Kincaid Grade Level Index~\cite{DBLP:conf/acl/NarayanG14}
def Flesch_Kincaid_grade_level(x, fkg_threshold, label):
  fkg = textstat.flesch_kincaid_grade(x.simp_text)

  if label == SIMPLE:
    if fkg <= fkg_threshold:
      return label
    else:
      return ABSTAIN
  else:
    if fkg > fkg_threshold:
      return label
    else:
      return ABSTAIN

def low_Flesch_Kincaid_grade_level(fkg_threshold, label):
  return LabelingFunction(
      name=f"low_fkg_threshold={fkg_threshold}",
      f=Flesch_Kincaid_grade_level,
      resources=dict(fkg_threshold=fkg_threshold, label=label), pre=[spacy_nlp]
  )


# christin: high Flesch reading ease~\cite{simpa}
def Flesch_Kincaid_reading_ease(x, fkre_threshold, label):
  fkre = textstat.flesch_reading_ease(x.simp_text)

  if label == SIMPLE:
    if fkre >= fkre_threshold:
      return label
    else:
      return ABSTAIN
  else:
    if fkre < fkre_threshold:
      return label
    else:
      return ABSTAIN

def high_Flesch_Kincaid_reading_ease(fkre_threshold, label):
  return LabelingFunction(
      name=f"high_fkre_threshold={fkre_threshold}",
      f=Flesch_Kincaid_reading_ease,
      resources=dict(fkre_threshold=fkre_threshold, label=label), pre=[spacy_nlp]
  )



# christin: no passive voice~\cite{arfe}
@labeling_function(pre=[spacy_nlp], name="no_passive_voice")
def lf_no_passive_voice(x):
  for sent in x.simp_sentences:
    passive_result = passivepy.match_text(sent, full_passive=True, truncated_passive=True)
    found = passive_result.binary[0] == 1

  if found:
    return ABSTAIN
  else:
    return SIMPLE

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

# christin: low sentence length (words)~\cite{arfe}, especially for children or non-native speakers~\cite{DBLP:conf/coling/StajnerH18}
def length_sents_max_thres(x, length_sent_threshold, label):
  num_words = []

  for sent in x.simp_doc.sents:
    len_sent_in_words = len([token for token in sent if token.pos_ != "PUNCT"])
    num_words.append(len_sent_in_words)

  final_num_words = max(num_words)

  if label == SIMPLE:
    if final_num_words <= length_sent_threshold:
      return label
    else:
      return ABSTAIN
  else:
    if final_num_words > length_sent_threshold:
      return label
    else:
      return ABSTAIN

def low_length_sents_max_thres(length_sent_threshold, label):
  return LabelingFunction(
      name=f"low_num_sents_max_thres={length_sent_threshold}",
      f=length_sents_max_thres,
      resources=dict(length_sent_threshold=length_sent_threshold, label=label),
      pre=[spacy_nlp]
  )

# christin: low sentence length (words)~\cite{arfe}, especially for children or non-native speakers~\cite{DBLP:conf/coling/StajnerH18}
def length_sents_avg_thres(x, length_sent_threshold, label):
  num_words = []

  for sent in x.simp_doc.sents:
    len_sent_in_words = len([token for token in sent if token.pos_ != "PUNCT"])
    num_words.append(len_sent_in_words)

  final_num_words = sum(num_words)/len(num_words)

  if label == SIMPLE:
    if final_num_words <= length_sent_threshold:
      return label
    else:
      return ABSTAIN
  else:
    if final_num_words > length_sent_threshold:
      return label
    else:
      return ABSTAIN

def low_length_sents_avg_thres(length_sent_threshold, label):
  return LabelingFunction(
      name=f"low_num_sents_avg_thres={length_sent_threshold}",
      f=length_sents_avg_thres,
      resources=dict(length_sent_threshold=length_sent_threshold, label=label),
      pre=[spacy_nlp]
  )

# christin: low number of sentences in text for people with intellectual disability~\cite{arfe}
def num_sents_num_thres(x, sent_num_threshold, label):
  num_sent = len(x.simp_sentences)

  if label == SIMPLE:
    if num_sent <= sent_num_threshold:
      return label
    else:
      return ABSTAIN
  else:
    if num_sent > sent_num_threshold:
      return label
    else:
      return ABSTAIN

def low_sents_num_thres(sent_num_threshold, label):
  return LabelingFunction(
      name=f"num_sents_num_thres={sent_num_threshold}",
      f=num_sents_num_thres,
      resources=dict(sent_num_threshold=sent_num_threshold, label=label),
      pre=[spacy_nlp]
  )

# christin: no conjunctions for people with language problems~\cite{arfe}
@labeling_function(pre=[spacy_nlp], name="no_conjunctions")
def lf_no_conjunctions(x):
  conj_pos = [token.pos_ for token in x.simp_doc if token.pos_ in ['CONJ', 'CCONJ', 'SCONJ']]

  if len(conj_pos) > 0:
    return ABSTAIN
  else:
    return SIMPLE

# christin: no conditional (if-then) clauses~\cite{arfe}
@labeling_function(pre=[spacy_nlp], name="no_conditional")
def lf_no_conditional(x):
  s_t = [sw.lower() for sw in x.simp_tokens]

  for con_word in [['if'], ['unless'], ['providing'], ['supposing'], ['suppose'], ['without'], ['provided', 'that'], ['as', 'long', 'as'], ['on', 'condition', 'that'], ['but', 'for']]:
    if con_word[0] in s_t:
      if len(con_word) == 1:
        return ABSTAIN
      else:
        cw_found = False
        for pos in [index for (index, item) in enumerate(s_t) if item == con_word[0]]:
          cw_pos = 1

          for cw in con_word[1:]:
            if pos + cw_pos < len(s_t) and s_t[pos + cw_pos] == cw:
              cw_found = True
            else:
              cw_found = False
            cw_pos += 1

          if cw_found:
            return ABSTAIN

  return SIMPLE

# christin: no appositions~\cite{DBLP:conf/acl/NarayanG14}
@labeling_function(pre=[spacy_nlp], name="no_apposition")
def lf_no_apposition(x):
  deps = [token.dep_ for token in x.simp_tokens_data]
  if 'appos' in deps:
    return ABSTAIN

  return SIMPLE

# christin grammatical correctness~\cite{DBLP:journals/tacl/XuCN15}
@labeling_function(pre=[spacy_nlp], name="no_grammatical_errors")
def lf_no_grammatical_errors(x):
  matches_us = tool_us.check(x.simp_text)
  matches_gb = tool_gb.check(x.simp_text)

  if len(matches_us) == 0 or len(matches_gb) == 0:
    return SIMPLE

  return ABSTAIN

# christin: fewer modifiers~\cite{DBLP:conf/acl/NarayanG14}
#@labeling_function(pre=[spacy_nlp], name="fewer_modifiers")
#def lf_fewer_modifiers(x):
#  mods = ['advmod', 'amod', 'nmod', 'npadvmod', 'nummod', 'quantmod']

#  deps_source = [token.dep_ for token in x.source_tokens_data if token.dep_ in mods]
#  deps_simp = [token.dep_ for token in x.simp_tokens_data if token.dep_ in mods]

#  if len(deps_simp) < len(deps_source):
#    return SIMPLE

#  return ABSTAIN

# christin: few NOT FEWER modifiers
def few_modifiers(x, few_mod_threshold, label):
  mods = ['advmod', 'amod', 'nmod', 'npadvmod', 'nummod', 'quantmod']

  deps_simp = [token.dep_ for token in x.simp_tokens_data if token.dep_ in mods]

  if label == SIMPLE:
    if len(deps_simp) <= few_mod_threshold:
      return label
    else:
      return ABSTAIN
  else:
    if len(deps_simp) > few_mod_threshold:
      return label
    else:
      return ABSTAIN

def few_modifiers_thres(few_mod_threshold, label):
  return LabelingFunction(
      name=f"few_modifiers_thres={few_mod_threshold}",
      f=few_modifiers,
      resources=dict(few_mod_threshold=few_mod_threshold, label=label),
      pre=[spacy_nlp]
  )

# christin: few noun phrases for people with poor language skills~\cite{arfe}
def few_noun_phrases(x, noun_phrase_thres, label):
  noun_phrases = [chunk.text for chunk in x.simp_doc.noun_chunks]

  if label == SIMPLE:
    if len(noun_phrases) <= noun_phrase_thres:
      return label
    else:
      return ABSTAIN
  else:
    if len(noun_phrases) > noun_phrase_thres:
      return label
    else:
      return ABSTAIN

def few_noun_phrases_thres(noun_phrase_thres, label):
  return LabelingFunction(
      name=f"few_noun_phrases_thres={noun_phrase_thres}",
      f=few_noun_phrases,
      resources=dict(noun_phrase_thres=noun_phrase_thres, label=label),
      pre=[spacy_nlp]
  )


def get_all_lfs():
  
  animals = ["dog", "fish", "sheep", "bunny", "octopus", "roadrunner", "okapi", "anisakis"]

  word_cnt_lfs_simple = [make_word_cnt_lf(w_cnt, label=SIMPLE) for w_cnt in range(3,15)]
  word_cnt_lfs_complex = [make_word_cnt_lf(w_cnt, label=NOT_SIMPLE) for w_cnt in range(15,30)]
  avg_concreteness_lfs_simple = [make_avg_conreteness_lf(threshold, label=SIMPLE) for threshold in np.round(np.linspace(2.5,4.5,5), 3)]
  avg_concreteness_lfs_complex = [make_avg_conreteness_lf(threshold, label=NOT_SIMPLE) for threshold in np.round(np.linspace(1.5,2.5,5), 3)]
  max_concreteness_lfs_simple = [make_max_conreteness_lf(threshold, label=SIMPLE) for threshold in np.round(np.linspace(3.5,4.5,5), 3)]
  max_concreteness_lfs_complex = [make_max_conreteness_lf(threshold, label=NOT_SIMPLE) for threshold in np.round(np.linspace(1.5,2.5,5), 3)]
  median_concreteness_lfs_simple = [make_median_conreteness_lf(threshold, label=SIMPLE) for threshold in np.round(np.linspace(2.5,4.5,5), 3)]
  median_concreteness_lfs_complex = [make_median_conreteness_lf(threshold, label=NOT_SIMPLE) for threshold in np.round(np.linspace(1.5,2.5,5), 3)]
  content_word_cnt_lfs_simple = [make_content_words_ratio_lf(ratio_threshold, label=SIMPLE) for ratio_threshold in np.round(np.linspace(0.01,0.3,10), 3)]
  content_word_cnt_lfs_complex = [make_content_words_ratio_lf(ratio_threshold, label=NOT_SIMPLE) for ratio_threshold in np.round(np.linspace(0.2,0.8,10), 3)]
  infrequent_words_lfs_simple = [make_infrequent_words_lf(p[0], p[1], label=SIMPLE) for p in [(a,b) for a in range(1,3) for b in animals]]
  infrequent_words_lfs_complex = [make_infrequent_words_lf(p[0], p[1], label=NOT_SIMPLE) for p in [(a,b) for a in range(2,6) for b in animals]]
  avg_aoa_lfs_simple = [make_avg_age_of_acquisition_lf(age, label=SIMPLE) for age in range(4,12)]
  avg_aoa_lfs_complex = [make_avg_age_of_acquisition_lf(age, label=NOT_SIMPLE) for age in range(8,18)]
  max_aoa_lfs_simple = [make_max_age_of_acquisition_lf(age, label=SIMPLE) for age in range(6,14)]
  max_aoa_lfs_complex = [make_max_age_of_acquisition_lf(age, label=NOT_SIMPLE) for age in range(10,20)]
  median_aoa_lfs_simple = [make_median_age_of_acquisition_lf(age, label=SIMPLE) for age in range(4,12)]
  median_aoa_lfs_complex = [make_median_age_of_acquisition_lf(age, label=NOT_SIMPLE) for age in range(8,18)]
  avg_image_lfs_simple = [make_avg_imageability_lf(imageability_threshold, label=SIMPLE) for imageability_threshold in [4.0, 4.2]]
  avg_image_lfs_complex = [make_avg_imageability_lf(imageability_threshold, label=NOT_SIMPLE) for imageability_threshold in [2.5,2.7]]
  med_image_lfs_simple = [make_med_imageability_lf(imageability_threshold, label=SIMPLE) for imageability_threshold in [4.0, 4.2]]
  med_image_lfs_complex = [make_med_imageability_lf(imageability_threshold, label=NOT_SIMPLE) for imageability_threshold in [2.5,2.7]]
  entity_token_ratio_text_lfs = [make_entity_token_ratio_text_lf(thresh, label=SIMPLE) for thresh in [0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]]
  entity_token_ratio_text_lfs_complex = [make_entity_token_ratio_text_lf(thresh, label=NOT_SIMPLE) for thresh in [0.6, 0.7, 0.8, 0.9, 1]]
  entity_token_ratio_sentence_lfs = [make_entity_token_ratio_sentence_lf(thresh, label=SIMPLE) for thresh in [0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]]
  entity_token_ratio_sentence_lfs_complex = [make_entity_token_ratio_sentence_lf(thresh, label=NOT_SIMPLE) for thresh in [0.6, 0.7, 0.8, 0.9, 1]]
  entity_token_ratio_paragraph_lfs = [make_entity_token_ratio_paragraph_lf(thresh, label=SIMPLE) for thresh in [0, 0.02, 0.05, 0.1, 0.2, 0.3]]
  entity_token_ratio_paragraph_lfs_complex = [make_entity_token_ratio_paragraph_lf(thresh, label=NOT_SIMPLE) for thresh in [0.4,0.5, 0.6, 0.7, 0.8, 0.9, 1]]
  num_unique_lemmas_lfs = [make_num_unique_lemmas_lf(thresh, label=SIMPLE) for thresh in [5, 10, 15, 20, 25]]
  num_unique_lemmas_complex = [make_num_unique_lemmas_lf(thresh, label=NOT_SIMPLE) for thresh in [30, 35, 40, 45, 50]]
  num_unique_lemmas_norm_lfs = [make_num_unique_lemmas_norm_lf(thresh, label=SIMPLE) for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]]
  num_unique_lemmas_norm_complex = [make_num_unique_lemmas_norm_lf(thresh, label=NOT_SIMPLE) for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]]
  avg_depth_of_syntactic_tree_lfs = [make_depth_of_syntactic_tree_lf(thresh, label=SIMPLE) for thresh in [1, 2, 3, 4]]
  avg_depth_of_syntactic_tree_complex = [make_depth_of_syntactic_tree_lf(thresh, label=NOT_SIMPLE) for thresh in [5, 6, 7, 8, 9, 10, 11, 12]]
  avg_num_punctuation_text_lfs = [make_avg_num_punctuation_text_lf(thresh, label=SIMPLE) for thresh in [1, 1.2, 1.5, 2, 2.5]]
  avg_num_punctuation_text_lfs_complex = [make_avg_num_punctuation_text_lf(thresh, label=NOT_SIMPLE) for thresh in [2.5, 3, 4]]
  unique_entities_text_lfs = [make_unique_entities_text_lf(thresh, label=SIMPLE) for thresh in [0, 1, 2, 3]]
  unique_entities_text_lfs_complex = [make_unique_entities_text_lf(thresh, label=NOT_SIMPLE) for thresh in [4, 5, 6, 7, 8, 9, 10]]
  average_entities_sentence_lfs = [make_average_entities_sentence_lf(thresh, label=SIMPLE) for thresh in [0.2, 0.5, 1, 2, 3]]
  average_entities_sentence_lfs_complex = [make_average_entities_sentence_lf(thresh, label=NOT_SIMPLE) for thresh in [4, 5, 6, 7, 8, 9, 10]]
  average_entities_paragraph_lfs = [make_average_entities_paragraph_lf(thresh, label=SIMPLE) for thresh in [0, 1, 2, 3, 4]]
  average_entities_paragraph_lfs_complex = [make_average_entities_paragraph_lf(thresh, label=NOT_SIMPLE) for thresh in [ 3, 4, 5, 6, 7, 8, 9, 10]]
  unique_entity_total_entity_ratio_text_lfs = [make_unique_entity_total_entity_ratio_text_lf(thresh, label=SIMPLE) for thresh in [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]
  unique_entity_total_entity_ratio_text_lfs_complex = [make_unique_entity_total_entity_ratio_text_lf(thresh, label=NOT_SIMPLE) for thresh in [0.9, 1]]
  unique_entity_total_entity_ratio_sentence_lfs = [make_unique_entity_total_entity_ratio_sentence_lf(thresh, label=SIMPLE) for thresh in [0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]
  unique_entity_total_entity_ratio_sentence_lfs_complex = [make_unique_entity_total_entity_ratio_sentence_lf(thresh, label=NOT_SIMPLE) for thresh in [0.6, 0.7, 0.8, 0.9, 1]]
  unique_entity_total_entity_ratio_paragraph_lfs = [make_unique_entity_total_entity_ratio_paragraph_lf(thresh, label=SIMPLE) for thresh in [0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4,0.5, 0.6,]]
  unique_entity_total_entity_ratio_paragraph_lfs_complex = [make_unique_entity_total_entity_ratio_paragraph_lf(thresh, label=NOT_SIMPLE) for thresh in [ 0.7, 0.8, 0.9, 1]]
  no_relative_clauses_lfs = [make_no_relative_clauses_lf(thresh, label=SIMPLE) for thresh in [0, 1, 2, 3]]
  no_relative_sub_clauses_lfs = [make_no_relative_sub_clauses_lf(thresh, label=SIMPLE) for thresh in [0, 1, 2, 3]]
  few_anaphors_lfs = [make_few_anaphors_lf(thresh, label=SIMPLE) for thresh in [0, 1]]
  distance_appearance_same_entities_paragraph_lfs = [make_distance_appearance_same_entities_paragraph_lf(thresh_distance, thresh_number, label=SIMPLE) for thresh_distance in [1, 2, 3] for thresh_number in [1,2,3]]
  avarage_distance_appearance_same_entities_paragraph_lfs = [make_average_distance_appearance_same_entities_paragraph_lf(thresh, label=SIMPLE) for thresh in [0.2, 0.5, 1, 2, 3, 4 ]]
  avarage_distance_appearance_same_entities_sentence_lfs = [make_average_distance_appearance_same_entities_sentence_lf(thresh, label=SIMPLE) for thresh in [ 0.2, 0.5, 1, 2, 3, 4, 5 ]]
  distance_appearance_same_entities_sentence_lfs = [make_distance_appearance_same_entities_sentence_lf(thresh_distance, thresh_number, label=SIMPLE) for thresh_distance in [1, 2, 3] for thresh_number in [1,2,3]]
  avarage_distance_entities_sentence_consec_lfs = [make_avarage_distance_entities_lf(thresh, scope="sent", same_or_consecutive="consec", label=SIMPLE) for thresh in [1,2,4,6,10]]
  avarage_distance_entities_sentence_same_lfs = [make_avarage_distance_entities_lf(thresh, scope="sent", same_or_consecutive="same", label=SIMPLE) for thresh in [1,2,4,6, 10]]
  avarage_distance_entities_paragraph_consec_lfs = [make_avarage_distance_entities_lf(thresh, scope="para", same_or_consecutive="consec", label=SIMPLE) for thresh in [2,4,8,16,32]]
  avarage_distance_entities_paragraph_same_lfs = [make_avarage_distance_entities_lf(thresh, scope="para", same_or_consecutive="same", label=SIMPLE) for thresh in [2,4,8,16,32]]
  lfs_proportions_of_long_words_syllables_simple = [low_proportion_of_long_words_syllables(long_length, proportion, label=SIMPLE) for long_length in (2, 3, 4) for proportion in (0.05, 0.1, 0.15, 0.2, 0.25)]
  lfs_proportions_of_long_words_letters_simple = [low_proportion_of_long_words_letters(long_length, proportion, label=SIMPLE) for long_length in (5, 6, 7, 8, 9) for proportion in (0.05, 0.1, 0.15, 0.2, 0.25)]
  lfs_low_fkg_simple = [low_Flesch_Kincaid_grade_level(fkg_threshold, label=SIMPLE) for fkg_threshold in (5, 6, 7, 8, 9)]
  lfs_high_fkre_simple = [high_Flesch_Kincaid_reading_ease(fkre_threshold, label=SIMPLE) for fkre_threshold in (100, 90, 80, 70, 60)]
  lfs_avg_Levenshtein = [low_avg_Levenshtein(lev_threshold, label=SIMPLE) for lev_threshold in (0.1, 0.2, 0.3, 0.4, 0.5)]
  lfs_low_length_sents_max = [low_length_sents_max_thres(length_sent_threshold, label=SIMPLE) for length_sent_threshold in (10, 15, 20)]
  lfs_low_length_sents_avg = [low_length_sents_avg_thres(length_sent_threshold, label=SIMPLE) for length_sent_threshold in (10, 15, 20, 25)]
  lfs_low_sents_num = [low_sents_num_thres(sent_num_threshold, label=SIMPLE) for sent_num_threshold in (1, 2, 3, 4, 5)]
  lfs_few_modifiers = [few_modifiers_thres(few_mod_threshold, label=SIMPLE) for few_mod_threshold in (0, 1, 2, 3, 4, 5)]
  lfs_few_noun_phrases = [few_noun_phrases_thres(noun_phrase_thres, label=SIMPLE) for noun_phrase_thres in (0, 1, 2, 3, 4, 5)]
  ratio_academic_word_list_lfs = [make_ratio_academic_word_list_lf(thresh, label=SIMPLE) for thresh in [0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.13]]
  ratio_academic_word_list_complex_lfs = [make_ratio_academic_word_list_lf(thresh, label=NOT_SIMPLE) for thresh in [0.14, 0.19, 0.25]]
  avg_num_words_before_main_verb_lfs = [make_avg_num_words_before_main_verb_lf(thresh, label=SIMPLE) for thresh in [1, 2, 3, 4, 6, 8]]
  avg_num_words_before_main_verb_complex_lfs = [make_avg_num_words_before_main_verb_lf(thresh, label=NOT_SIMPLE) for thresh in [10, 12, 15]]
  perc_past_perfect_lfs = [make_perc_past_perfect_lf(thresh, label=SIMPLE) for thresh in [0, 0.1, 0.2, 0.4, 0.6, 0.8]]
  perc_past_perfect_complex_lfs = [make_perc_past_perfect_lf(thresh, label=NOT_SIMPLE) for thresh in [0.6, 0.8, 1]]
  num_past_perfect_lfs = [make_num_past_perfect_lf(thresh, label=SIMPLE) for thresh in [0, 1, 2, 3, 4]]
  num_past_perfect_complex_lfs = [make_num_past_perfect_lf(thresh, label=NOT_SIMPLE) for thresh in [5, 6, 7, 8, 12, 15]]
  perc_past_tense_lfs = [make_perc_past_tense_lf(thresh, label=SIMPLE) for thresh in [0, 0.1, 0.2, 0.4, 0.6, 0.8]]
  perc_past_tense_complex_lfs = [make_perc_past_tense_lf(thresh, label=NOT_SIMPLE) for thresh in [0.6, 0.8, 1]]
  num_past_tense_lfs = [make_num_past_tense_lf(thresh, label=SIMPLE) for thresh in [0, 1, 2, 3, 4]]
  num_past_tense_complex_lfs = [make_num_past_tense_lf(thresh, label=NOT_SIMPLE) for thresh in [5, 6, 7, 8, 12, 15]]
  freq_third_person_singular_pronouns_lfs = [make_freq_third_person_singular_pronouns_lf(thresh, label=SIMPLE) for thresh in [0, 1, 2]]
  freq_third_person_singular_pronouns_lfs_complex = [make_freq_third_person_singular_pronouns_lf(thresh, label=NOT_SIMPLE) for thresh in [1, 2, 3, 4, 5]]
  freq_negations_lfs = [make_freq_negations_lf(thresh, label=SIMPLE) for thresh in [0, 1, 2]]
  freq_negations_lfs_complex = [make_freq_negations_lf(thresh, label=NOT_SIMPLE) for thresh in [1, 2, 3, 4, 5]]
  freq_nominalisations_lfs = [make_freq_nominalisations_lf(thresh, label=SIMPLE) for thresh in [0, 1, 2]]
  freq_nominalisations_lfs_complex = [make_freq_nominalisations_lf(thresh, label=NOT_SIMPLE) for thresh in [1, 2, 3, 4, 5]]
  perc_more_than_8_characters_lfs = [make_perc_more_than_8_characters_lf(thresh, label=SIMPLE) for thresh in [0, 0.02, 0.05, 0.1, 0.2, 0.3]]
  perc_more_than_8_characters_complex_lfs = [make_perc_more_than_8_characters_lf(thresh, label=NOT_SIMPLE) for thresh in [0.25, 0.3, 4]]
  perc_vocab_initial_forLang_learn_lfs = [make_perc_vocab_initial_forLang_learn_lf(thresh, label=SIMPLE) for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]]
  perc_vocab_initial_forLang_learn_lfs_complex = [make_perc_vocab_initial_forLang_learn_lf(thresh, label=NOT_SIMPLE) for thresh in [0.6, 0.7, 0.8, 0.9, 1]]


  
  # [lf_fewer_modifiers] temp out


  all_lfs = word_cnt_lfs_simple + word_cnt_lfs_complex + infrequent_words_lfs_simple + infrequent_words_lfs_complex + entity_token_ratio_text_lfs + \
            entity_token_ratio_text_lfs_complex + lfs_proportions_of_long_words_syllables_simple + lfs_proportions_of_long_words_letters_simple + lfs_low_fkg_simple + \
            lfs_high_fkre_simple + [lf_no_passive_voice] + lfs_avg_Levenshtein + median_aoa_lfs_simple + median_aoa_lfs_complex + max_aoa_lfs_simple + \
            max_aoa_lfs_complex + avg_aoa_lfs_simple + avg_aoa_lfs_complex + median_concreteness_lfs_complex + max_concreteness_lfs_simple + \
            max_concreteness_lfs_complex + avg_concreteness_lfs_simple + avg_concreteness_lfs_complex + unique_entity_total_entity_ratio_paragraph_lfs + \
            unique_entity_total_entity_ratio_sentence_lfs + unique_entity_total_entity_ratio_text_lfs + average_entities_paragraph_lfs + average_entities_sentence_lfs + \
            unique_entities_text_lfs + entity_token_ratio_paragraph_lfs + entity_token_ratio_sentence_lfs + lfs_low_length_sents_max + lfs_low_length_sents_avg + lfs_low_sents_num + \
            [lf_no_conjunctions] + [lf_no_conditional] + [lf_no_apposition] + [lf_no_grammatical_errors] + distance_appearance_same_entities_paragraph_lfs + \
            lfs_few_modifiers + distance_appearance_same_entities_sentence_lfs + avarage_distance_appearance_same_entities_sentence_lfs + \
            lfs_few_noun_phrases + avg_image_lfs_simple + avg_image_lfs_complex + med_image_lfs_simple + med_image_lfs_complex +avarage_distance_entities_sentence_consec_lfs +\
            avarage_distance_entities_sentence_same_lfs+ avarage_distance_entities_paragraph_consec_lfs + avarage_distance_entities_paragraph_same_lfs + \
            avg_depth_of_syntactic_tree_lfs + avg_depth_of_syntactic_tree_complex + avg_num_punctuation_text_lfs + avg_num_punctuation_text_lfs_complex + \
            num_unique_lemmas_lfs + num_unique_lemmas_complex + num_unique_lemmas_norm_lfs + num_unique_lemmas_norm_complex + ratio_academic_word_list_lfs + \
            ratio_academic_word_list_complex_lfs + median_concreteness_lfs_simple + content_word_cnt_lfs_simple + content_word_cnt_lfs_complex + entity_token_ratio_sentence_lfs_complex +\
            entity_token_ratio_paragraph_lfs_complex + unique_entities_text_lfs_complex + average_entities_sentence_lfs_complex + average_entities_paragraph_lfs_complex +\
            unique_entity_total_entity_ratio_text_lfs_complex + unique_entity_total_entity_ratio_sentence_lfs_complex+ unique_entity_total_entity_ratio_paragraph_lfs_complex +\
            no_relative_clauses_lfs + no_relative_sub_clauses_lfs + few_anaphors_lfs + avarage_distance_appearance_same_entities_paragraph_lfs + \
            avg_num_words_before_main_verb_lfs + avg_num_words_before_main_verb_complex_lfs +perc_past_perfect_lfs + perc_past_perfect_complex_lfs +\
            num_past_perfect_lfs + num_past_perfect_complex_lfs + perc_past_tense_lfs + perc_past_tense_complex_lfs + num_past_tense_lfs + num_past_tense_complex_lfs +\
            freq_third_person_singular_pronouns_lfs + freq_third_person_singular_pronouns_lfs_complex + freq_negations_lfs + freq_negations_lfs_complex +\
            freq_nominalisations_lfs + freq_nominalisations_lfs_complex + perc_more_than_8_characters_lfs + perc_more_than_8_characters_complex_lfs +\
            perc_vocab_initial_forLang_learn_lfs + perc_vocab_initial_forLang_learn_lfs_complex
  return all_lfs