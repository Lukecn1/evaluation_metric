import json
import pandas as pd
import numpy as np
import nltk

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.util import ngrams
from nltk import skipgrams

# Sets the danish stemmer from the nltk-snowballstemmer
stemmer = SnowballStemmer('english')

# List of stopwords
stopwords_ = set(stopwords.words('english'))
stopwords_.update([',', '.','-', ';','\n','\r', '\xa0', '.'])



###################### EVALUATION FUNCTIONS ######################

# Returns the n-grams for a given string
# Note: each word is stemmed to its root form
def get_ngrams(summary, n, remove_stopwords):

    n_grams = list()

    tokens = nltk.word_tokenize(summary)

    if remove_stopwords:
        tokens = [stemmer.stem(w1) for w1 in [w.lower() for w in tokens if w not in stopwords_]]
    else:
        tokens = [stemmer.stem(w1) for w1 in [w.lower() for w in tokens]]

    if n == 1:
        return tokens
    
    return n_grams + list(ngrams(tokens, n))



# ROUGE-N for n grams. 
# Returns the F-score for the given summary
# Objective funciton to maximize in selection process
def rouge_N(reference_summary, candidate_summary, n, remove_stopwords):

    set_ext = set(get_ngrams(candidate_summary, n, remove_stopwords))
    set_abs = set(get_ngrams(reference_summary, n, remove_stopwords))

    common_ngrams = set_ext & set_abs

    if len(set_ext) < 1 or len(set_abs) < 1:
        return 0.0, 0.0, 0.0
    
    precision = len(common_ngrams)/len(set_ext)
    recall = len(common_ngrams)/len(set_abs)

    if(recall > 0 and precision > 0):
        f_score  = 2 * (recall*precision) / (recall + precision)
    else:
        f_score = 0.0

    return f_score



# Returns the length of the longest common subsequence between two strings.
def get_union_LCS(ref_sent, cand_sent, remove_stopwords):

    ref_tokens = get_ngrams(ref_sent, 1, remove_stopwords) 
    cand_tokens = get_ngrams(cand_sent, 1, remove_stopwords)

    indicies = list()

    prev_match_index = -1 # Used to ensure sentence word order is maintained when calculating LCS
    
    for i in range(len(ref_tokens)):
        for j in range(len(cand_tokens)):
            if ref_tokens[i] == cand_tokens[j] and prev_match_index < j:
                prev_match_index = j
                indicies.append(i)

    return indicies



# Returns the union LCS score for a given refernece summary sentence
# Is used in the calcualtion of the summary level LCS
def union_lcs(ref_sentence, candidate_summary, remove_stopwords):

    lcs_u = list()

    for cand_sent in candidate_summary:
        lcs_u.append(get_union_LCS(ref_sentence, cand_sent, remove_stopwords))

    lcs_flattened = []
    for sublist in lcs_u:
        for index in sublist:
            lcs_flattened.append(index)

    return set(lcs_flattened)    



# ROUGE-L for two summaries
# Uses the concept of (longest-common-subsequence) to calcualte the score
def rouge_L(reference_summary, candidate_summary, remove_stopwords):

    final_lcs = 0 

    reference_sentences = nltk.sent_tokenize(reference_summary, language = 'english')
    candidate_sentences = nltk.sent_tokenize(candidate_summary, language = 'english')
    
    for ref_sent in reference_sentences:
        final_lcs += len(union_lcs(ref_sent, candidate_sentences, remove_stopwords))

    if len(reference_summary) < 1 or len(candidate_summary) < 1:
        return 0.0 
    
    recall = final_lcs / len(get_ngrams(reference_summary, 1, remove_stopwords)) 
    precision = final_lcs / len(get_ngrams(candidate_summary, 1, remove_stopwords)) 
    f_score = 0.0

    if recall > 0 and precision > 0:
        f_score = 2.0 * ((precision * recall) / (precision + recall)) 
    else:
        f_score = 0.0
        
    return f_score