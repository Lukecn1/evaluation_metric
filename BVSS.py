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




def gen_bert_vectors(candidate_summary, reference_summary):
    """
    Generates the embedding vectors for the given sentences/tokens
    Uses the BERT as Service Client to produce the vectors. 

    Args:
        - :param: `candidate_summary` (list of str): candidate summary sentences
        - :param: `reference_summary` (list of str): reference summary sentences
    
    Return:
        - :param: candidate_vectors (list of lists of float): matrix of vectors for the candidate summary
        - :param: reference_vectors (list of lists of float): matrix of vectors for the reference summary
    """

    candidate_vectors = bert_client.encode(candidate_summary)
    reference_vectors = bert_client.encode(reference_summary)
    
    return candidate_vectors, reference_vectors



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
    
    precision_scores = []
    recall_scores = []
    f1_scores = []

    if scoring_approach == 'argmax':
        for cand_sent in cand_sentences:
            p, r, f1 = bert_score.score([cand_sent], [ref_sentences], model_type = model, num_layers = layer, lang = language) # BERTscore defaults to taking the argmax value when multiple references are given for 1 candidate sentence
            precision_scores.append(p.tolist()[0])
            recall_scores.append(r.tolist()[0])
            f1_scores.append(f1.tolist()[0])

    elif scoring_approach == 'mean':
        for cand_sent in cand_sentences:
            for ref_sent in ref_sentences:
                p, r, f1 = bert_score.score([cand_sent], [ref_sent], model_type = model, num_layers = layer, lang = language)  
                precision_scores.append(p.tolist()[0])
                recall_scores.append(r.tolist()[0])
                f1_scores.append(f1.tolist()[0])
    
    else:
      print("scoring_approach parameter must be defined as either 'argmax' or 'mean'. Check the README for descriptions of each.")

    precision_mean = sum(precision_scores)  / len(precision_scores)
    recall_mean = sum(recall_scores)  / len(recall_scores)
    f1_mean = sum(f1_scores)  / len(f1_scores)

    return precision_mean, recall_mean, f1_mean


#cand_test = ["The cat in the hat got hit with the bat.", "The man got sad and sat on his back."]
#ref_test = ["The story tells the sad tale of the hat-wearing feline that meets it's demise with a strike of a wooden plank.", "The perpetrator riddled with sadness sat down.", "His further antics are not described"]
#get_bertscore(cand_test, ref_test, 'bert-base-uncased', 9, 'en', 'argmax')


def get_bvss(cand_sentences, ref_sentences, model, layer, language, scoring_approach):

  """
    BVSS metric

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


    precision_scores = []
    recall_scores = []
    f1_scores = []

    if scoring_approach == 'argmax':
        for cand_sent in cand_sentences:
            for ref_sent in ref_sentences:



    precision_mean = sum(precision_scores)  / len(precision_scores)
    recall_mean = sum(recall_scores)  / len(recall_scores)
    f1_mean = sum(f1_scores)  / len(f1_scores)

    return precision_mean, recall_mean, f1_mean
