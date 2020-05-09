import pandas as pd 
import numpy as np
import torch
import transformers
import bert_score
import nltk

from bert_serving.client import BertClient
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
#from sklearn.decomposition import pca
from scipy.spatial.distance import cosine

#df_tac = pd.read_csv('C:/Users/Lukas/ITU/Master_Thesis/Metric_paper/Data/TAC/TAC_ROUGE-HUMAN-REF.csv', sep= '\t')

bert_client = BertClient(ip = 'localhost', check_length = False)

# print(df_tac.columns)


###### HELPER FUNCTIONS ####### 


def tokenize_summaries(ref_sum, cand_sum, language, n = None):
  """
  Parameters:
  ref_sum, cand_sum: strings
  language: string
  n: tokenization level (if left "None" level is sentence)

  returns: Lists of tokens/n-grams  
  """

  if n == None:
    ref_sum_tokens = nltk.sent_tokenize(ref_sum, language = language)
    cand_sum_tokens = nltk.sent_tokenize(cand_sum, language = language)

  else:
    ref_sum_tokens = list(nltk.ngrams(nltk.word_tokenize(ref_sum), n))
    cand_sum_tokens = list(nltk.ngrams(nltk.word_tokenize(cand_sum), n))

  return ref_sum_tokens, cand_sum_tokens




def gen_bert_vectors(ref_sum, cand_sum):
    """
    Generates the embedding vectors for the given sentences/tokens
    Uses the BERT as Service Client to produce the vectors. 

    Inputs are assumed to be lists of tokens for the reference and candidate summaries respectively. 
    Output are lists of vectors (lists of lists)
    """

    reference_vectors = bert_client.encode(ref_sum)
    candidate_vectors = bert_client.encode(cand_sum)
    
    return reference_vectors, candidate_vectors



###### SCORING FUNCTIONS ####### 


def get_bertscore(cand_sentences, ref_sentences, model, layer, language, scoring_approach):
    """
    BERTScore metric, from the paper https://arxiv.org/pdf/1904.09675.pdf

    Args:
        - :param: `cand_sentences` (list of str): candidate summary sentences
        - :param: `ref_sentences` (list of str): reference summary sentences
        - :param: `model` (str): the specific bert model to use
        - :param: `Layer` (int): the layer of representation to use.
        - :param: `language` (str): language of the inputs.
                   performance may vary for non-english langauges on english pre-trained bert models     
        - :param: `scoring_approach` (str): defines whether to use the argmax or mean-based scoring approaches.
                   argmax returns the score of the highest scoring reference sentence for each candidate sentence 
                   mean-based returns the mean of all reference sentence scores for each candidate sentence 

    Return:
        - :param: precision score (float): precision score for the candidate summary 
        - :param: recall score (float): recall score for the candidate summary 
        - :param: f1 score (float): f1 score for the candidate summary 
    """

    """
    KEY ISSUE TO SOLVE HERE: 1) SCORE FOR EACH CANDIDATE SENTENCE
                             2) MEAN-BASED SCORING STRATEGY (take mean at sentence level rather than token level- keeps the original idea of BERTscore intact whlie adjusting the downstream scoring)                               
    """

    if scoring_approach != 'argmax' or scoring_approach != 'mean':
      return("scoring_approach must be defined as either 'argmax' or 'mean'. Check README for descriptions of each.")

    if scoring_approach == 'argmax':
        f1_scores = []
        for cand_sent in cand_sentences:
            p, r, f1 = bert_score.score([cand_sent], [ref_sentences], model_type = model, num_layers = layer, lang = language)
            f1_scores.append(f1.tolist()[0])
        
        recall = sum(f1_scores)/len(ref_sentences)
        precision = sum(f1_scores)/len(cand_sentences)

        final_mean_score = sum(f1_scores)  / len(f1_scores)
        final_f1_score = 2 * ( (recall * precision) / (recall + precision) )
    
    elif scoring_approach == 'mean':
        "define mean based scoring metric"


    final_max_scores = []
    for cand_sent in cand_sentences:
      f1_scores = []
      for ref_sent in ref_sentences:
        p, r, f1 = bert_score.score([cand_sent], [ref_sent], model_type = model, num_layers = layer, lang = language)  
      #  precision_scores = precision_Scores.append(p)
      # recall_scores = recall_Scores.append(r)
        f1_scores.append(f1.tolist()[0])
      final_max_scores.append(max(f1_scores))  
      
    print(final_max_scores)

cand_test = ["The cat in the hat got hit with the bat.", "The man got sad and sat on his back."]
ref_test = ["The story tells the sad tale of the hat-wearing feline that meets it's demise with a strike of a wooden plank.", "The perpetrator riddled with sadness sat down.", "His further antics are not described"]

get_bertscore(cand_test, ref_test, 'bert-base-uncased', 9, 'en', 'argmax')


def get_bvss():

  """
    BVSS metric

    Args:
        - :param: `cand_tokens` (list of str): candidate tokens
        - :param: `ref_tokens` (list of str): reference tokens
        - :param: `model` (str): the specific bert model to use
        - :param: `Layer` (int): the layer of representation to use.
        - :param: `language` (str): language of the inputs.
                   performance may vary for non-english langauges on english pre-trained bert models                  

    Return:
        - :param: f1 score (float): bvss f1 score for the candidate summary given the reference summary
   """
  pass

