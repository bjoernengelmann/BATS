import spacy
import spacy_universal_sentence_encoder
from bert_score import score
import warnings
import logging
import transformers
transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)



#sent_encoder = spacy_universal_sentence_encoder.load_model('en_use_lg')
sent_encoder = spacy_universal_sentence_encoder.load_model('en_use_md')
nlp = spacy.load('en_core_web_sm')

# Stopword removal
stop_words = spacy.lang.en.stop_words.STOP_WORDS
def remove_stop_words(sentence):
  doc = nlp(sentence)

  filtered_tokens = [token for token in doc if not token.is_stop]
  return ' '.join([token.text for token in filtered_tokens])


def check_sent_sim(s1, s2):
  return sent_encoder(s1).similarity(sent_encoder(s2))

def find_term_with_highest_sim(s1, s2):
  best = ("", 0)
  sim_base = check_sent_sim(s1, s2)
  for w in [tok.text for tok in nlp(s2)]:
    curr_sent = s2.replace(w, "")
    curr_sim_change = check_sent_sim(s1, curr_sent)-sim_base
    if curr_sim_change > best[1]:
      best = (w, curr_sim_change)
  return best

def core_preserved_meaning(s1, s2):
  best_w = find_term_with_highest_sim(s1, s2)
  if best_w and best_w[1] > 0:
    return core_preserved_meaning(s1, s2.replace(best_w[0],"").replace("  ", " "))
  else:
    return check_sent_sim(s1, s2), s2

def core_preserved_meaning_max_depth_5(s1, s2, depth=0):
  best_w = find_term_with_highest_sim(s1, s2)
  if best_w and best_w[1] > 0:
    if depth <=4:
      return core_preserved_meaning_max_depth_5(s1, s2.replace(best_w[0],"").replace("  ", " "), depth+1)
    else:
      return check_sent_sim(s1, s2), s2, depth
  else:
    return check_sent_sim(s1, s2), s2, depth
  
def mpire_score(s1, s2):
  return core_preserved_meaning_max_depth_5(s1, s2)[0]
  
#https://github.com/Tiiiger/bert_score
def bertscore(s1, s2):
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    P, R, F1 = score([s1], [s2], lang="en", verbose=False, rescale_with_baseline=True)
    return P[0].item()
