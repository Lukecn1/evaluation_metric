import pandas as pd 
import numpy as np
import torch
import nltk
import transformers
import bert_score
from bert_serving.client import BertClient
from bert_serving.server.helper import get_args_parser, get_shutdown_parser
from bert_serving.server import BertServer
from scipy.spatial.distance import cosine

from utils import launch_bert_as_service_server
from encode import get_embedding_vectors

"""
df_tac = pd.read_csv('C:/Users/Lukas/ITU/Master_Thesis/Metric_paper/Data/TAC/TAC_ROUGE-HUMAN-REF.csv', sep= '\t')
print(df_tac['Summary'])
print(df_tac['refsum1'])
print(df_tac['refsum2'])
print(df_tac['refsum3'])
print(df_tac['refsum4'])
"""


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
    
    final_precision_scores = []
    final_recall_scores = []
    final_f1_scores = []

    if scoring_approach == 'argmax':
        for cand_sent in cand_sentences:
            p, r, f1 = bert_score.score([cand_sent], [ref_sentences], model_type = model, num_layers = layer, lang = language) # BERTscore defaults to taking the argmax value when multiple references are given for 1 candidate sentence
            final_precision_scores.append(p.tolist()[0])
            final_recall_scores.append(r.tolist()[0])
            final_f1_scores.append(f1.tolist()[0])

    elif scoring_approach == 'mean':
        for cand_sent in cand_sentences:
            precision_scores = 0.0
            recall_scores = 0.0
            f1_scores = 0.0
            for ref_sent in ref_sentences:
                p, r, f1 = bert_score.score([cand_sent], [ref_sent], model_type = model, num_layers = layer, lang = language)  # BERTscore is the argmax of each word-comparision, we take the mean of the total argmax score for each candidate sentence
                precision_scores += p.tolist()[0]
                recall_scores += r.tolist()[0]
                f1_scores += f1.tolist()[0]
            
            # Divide with len(ref_sentences) to get the mean BERTscore for each candidate sentence
            final_precision_scores.append(precision_scores / len(ref_sentences))
            final_recall_scores.append(recall_scores / len(ref_sentences))
            final_f1_scores.append(f1_scores / len(ref_sentences))
    
    else:
        print("scoring_approach parameter must be defined as either 'argmax' or 'mean'. Check the README for descriptions of each.")
        return None

    # Final score is simply the average of the precision, recall and f1 score of each sentence in the candidate summary
    precision_score = sum(final_precision_scores)  / len(final_precision_scores)
    recall_score = sum(final_recall_scores)  / len(final_recall_scores)
    f1_score = sum(final_f1_scores)  / len(final_f1_scores)

    return precision_score, recall_score, f1_score


#cand_test = ["The cat in the hat got hit with the bat.", "The man got sad and sat on his back."]
#ref_test = ["The story tells the sad tale of the hat-wearing feline that meets it's demise with a strike of a wooden plank.", "The perpetrator riddled with sadness sat down.", "His further antics are not described"]
#get_bertscore(cand_test, ref_test, 'bert-base-uncased', 9, 'en', 'argmax')


def get_bvss(candidate_vectors, reference_vectors, scoring_approach):
    """
    BVSS metric

    Args:
        - :param: `candidate_vectors` (list of list of float): candidate summary embedding vectors
        - :param: `ref_sentences` (list of list of float): reference summary embedding vectors
                  performance may vary for non-english langauges on english pre-trained bert models     
        - :param: `scoring_approach` (str): defines whether to use the argmax or mean-based scoring approaches.
                  argmax returns the score of the highest scoring reference sentence for each candidate sentence 
                  mean-based returns the mean of all reference sentence scores for each candidate sentence 

    Return:
        - :param: precision score (float): precision score for the candidate summary 
        - :param: recall score (float): recall score for the candidate summary 
        - :param: f1 score (float): f1 score for the candidate summary 
    """

    if scoring_approach != 'argmax' or scoring_approach != 'mean':
        print("scoring_approach parameter must be defined as either 'argmax' or 'mean'. Check the README for descriptions of each.")
        return None

    final_cosines = []

    for cand_vec in candidate_vectors:
        cosines = []
        for ref_vec in reference_vectors:
            cosines.append( 1 - cosine(cand_vec, ref_vec) )

        if scoring_approach == 'argmax':
            final_cosines.append(max(cosines))
        
        if scoring_approach == 'mean':
            final_cosines.append(sum(cosines) / len(cosines))
    
    cosine_sum = sum(cosines)

    precision = cosine_sum  / len(candidate_vectors)
    recall = cosine  / len(reference_vectors)
    f1 = 2 * (precision * recall) / (precision + recall) 

    return precision, recall, f1