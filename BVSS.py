import pandas as pd 
import numpy as np
import nltk
import bert_score
from bert_serving.client import BertClient
from bert_serving.server.helper import get_args_parser, get_shutdown_parser
from bert_serving.server import BertServer
from sklearn.metrics.pairwise import cosine_similarity

from utils import launch_bert_as_service_server, terminate_server
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
        - :param: `reference_vectors` (list of list of float): reference summary embedding vectors
        - :param: `scoring_approach` (str): defines whether to use the argmax or mean-based scoring approaches.
                  
    Return:
        - :param: precision score (float): precision score for the candidate summary 
        - :param: recall score (float): recall score for the candidate summary 
        - :param: f1 score (float): f1 score for the candidate summary 
    """
    precision_scores = []
    recall_scores = []
    f1_scores = []

    if scoring_approach == 'argmax' or scoring_approach == 'mean':

        for cand_vec in candidate_vectors:
            cosines = []
            for ref_vec in reference_vectors:
                cosines.append( cosine_similarity([cand_vec], [ref_vec])[0][0] ) # Matrix format is due to expected shape of sklearn metric

            if scoring_approach == 'argmax':
                precision_scores.append(max(cosines))
            
            if scoring_approach == 'mean':
                precision_scores.append(sum(cosines) / len(cosines))

            print('cand', cosines)
    
        for ref_vec in reference_vectors:
            cosines = []
            for cand_vec in candidate_vectors:
                cosines.append( cosine_similarity([cand_vec], [ref_vec])[0][0] ) # Matrix format is due to expected shape of sklearn metric

            if scoring_approach == 'argmax':
                recall_scores.append(max(cosines))
            
            if scoring_approach == 'mean':
                recall_scores.append(sum(cosines) / len(cosines))

            print('ref', cosines)
    else:
        print("scoring_approach parameter must be defined as either 'argmax' or 'mean'. Check the README for descriptions of each.")
        return None
    
    #print('Final Cosines: ', final_cosines)

    precision = sum(precision_scores)  / len(candidate_vectors)
    recall = sum(recall_scores)  / len(reference_vectors)
    f1 = 2 * ( (precision * recall) / (precision + recall) ) 

    return precision, recall, f1



def get_bvss_scores(candidate_summaries, reference_summaries, scoring_approach, model, layer, n_gram_encoding = None, pooling_strategy = None, pool_word_pieces = False, language = 'english'):
    """
    Returns the BVSS scores for each of the summary pairs and return them in a list. 

    Args:
        - :param: `candidate_summaries` (list of str): candidate summaries - each string is a sumary 
        - :param: `reference_summaries` (list of str): reference summaries - each string is a summary
        - :param: `scoring_approach`    (str): defines whether to use the argmax or mean-based scoring approaches.
        - :param: `model`               (str): name of the model to produce the embedding vectors.
        - :param: `layer`               (int): the model layer used to retrieve the embedding vectors.
        - :param  'n_gram_encoding'     (int): n-gram encoding level - desginates how many word vectors to combine for each final embedding vector
                                               defaults to none resulting in sentence-level embedding vectors
        - :param: `pooling_strategy`    (str): the vector combination strategy - used when 'n_gram_encoding' == 'None' 
                                               if 'None' -> embedding level defaults to the sentence level of each individual sentence
        - :param: `pool_word_pieces`    (bool): whether or not to preemptively pool word pieces when doing n-gram pooling
                                                only relevant when n_gram_encoding is not None i.e. not for sentence level vectors
        - :param: `language`            (str): the language of the summaries, used for tokenizing the sentences - defaults to english
       
    Return:
        - :param: precision scores (list of float): precision scores for the candidate summaries 
        - :param: recall scores    (list of float): recall scores for the candidate summaries 
        - :param: f1 scores        (list of float): f1 scores for the candidate summaries 
    """
    launch_bert_as_service_server(model, layer, n_gram_encoding, pooling_strategy)

    precision_scores = []
    recall_scores = []
    f1_scores = []

    candidate_summaries_sentences = []
    reference_summaries_sentences = []

    # Returns each summary as a list of its sentences
    for i in range(len(candidate_summaries)):
        candidate_summaries_sentences.append(nltk.sent_tokenize(candidate_summaries[i], language= language))
        reference_summaries_sentences.append(nltk.sent_tokenize(reference_summaries[i], language= language))
    
    print(candidate_summaries_sentences)
    print(reference_summaries_sentences)

    for i, _ in enumerate(candidate_summaries_sentences, 0):
        print(get_bertscore(candidate_summaries_sentences[i], reference_summaries_sentences[i], 'bert-base-uncased', 11, 'english', 'mean'))

    candidate_embeddings, reference_embeddings = get_embedding_vectors(candidate_summaries_sentences, 
                                                                       reference_summaries_sentences, 
                                                                       pool_word_pieces, 
                                                                       n_gram_encoding)

    for i in range(len(candidate_embeddings)):
        p, r, f1 = get_bvss(candidate_embeddings[i], reference_embeddings[i], scoring_approach)

        precision_scores.append(p)
        recall_scores.append(r)
        f1_scores.append(f1)

    terminate_server()

    return precision_scores, recall_scores, f1_scores



candidate_summaries = ['First candidate summary for testing. Another sentence for testing purposes. The final phrase is written here.', 
                        'Second candidate summary is written here. It only consists of two sentences.', 
                        'The third and final candidate summary is here. It has more than two sentences. Hence the third text sequence.'
                        ]

reference_summaries = [ 'Here is the first sentence of the reference summary. Only two individual sentences for this summary.', 
                        'Start of the second reference. Testing the controlflow of the embedding functions.',
                        'Lastly a single sentence reference summary.'
                        ]

nr_summaries = len(candidate_summaries)

precision_scores, recall_scores, f1_scores = get_bvss_scores(candidate_summaries, 
                                                             reference_summaries, 
                                                             scoring_approach= 'argmax', 
                                                             model = 'bert-base-uncased', 
                                                             layer= 11, 
                                                             n_gram_encoding= None, 
                                                             pooling_strategy= 'REDUCE_MEAN', 
                                                             pool_word_pieces= False, 
                                                             language= 'english')
    

print(precision_scores)
print(recall_scores)
print(f1_scores)
