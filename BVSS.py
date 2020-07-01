import pandas as pd 
import numpy as np
import nltk
import bert_score
from sklearn.metrics.pairwise import cosine_similarity
from utils import get_bert_model
from encode import get_embedding_vectors



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




def get_bvss(candidate_vectors, reference_vectors, scoring_approach):
    """
    Calculates the metric score for the given candidate summary wrt. the reference summary

    Args:
        - :param: `candidate_vectors` (list of list of float): candidate summary embedding vectors 
        - :param: `reference_vectors` (list of list of float): reference summary embedding vectors
        - :param: `scoring_approach` (str): defines whether to use the argmax or mean-based scoring approaches.
                  
    Return:
        - :param: precision score (float): precision score for the candidate summary 
        - :param: recall score (float): recall score for the candidate summary 
        - :param: f1 score (float): f1 score for the candidate summary 
    """

    scores = np.zeros( ( len(candidate_vectors), len(reference_vectors) ) )

    precision_scores = []
    recall_scores = []

    if scoring_approach == 'argmax' or scoring_approach == 'mean':

        for i, cand_vec in enumerate(candidate_vectors, 0):
            for j, ref_vec in enumerate(reference_vectors, 0):
                scores[i][j] = cosine_similarity([cand_vec], [ref_vec])[0][0]

        print('SCORES')
        print(scores)

        for i, _ in enumerate(candidate_vectors, 0):

            if scoring_approach == 'argmax':
                precision_scores.append(max(scores[i, :]))
                
            if scoring_approach == 'mean':
                cosines = scores[i, :].tolist()
                precision_scores.append( sum(cosines) / len(cosines) )

        for i, _ in enumerate(reference_vectors, 0):

            if scoring_approach == 'argmax':
                recall_scores.append(max(scores[:, i]))
                
            if scoring_approach == 'mean':
                cosines = scores[:, i].tolist()
                recall_scores.append( sum(cosines) / len(cosines) )
    else:

        print("scoring_approach parameter must be defined as either 'argmax' or 'mean'. Check the README for descriptions of each.")
        return None
    
    precision = sum(precision_scores)  / len(candidate_vectors)
    recall = sum(recall_scores)  / len(reference_vectors)

    #print('precision: ', precision)
    #print('recall: ', recall)

    f1 = 2 * ( (precision * recall) / (precision + recall) ) 

    return precision, recall, f1
    


def get_bvss_scores(candidate_summaries, reference_summaries, scoring_approach, model_name, layer, n_gram_encoding = None, pool_word_pieces = False, language = 'english'):
    """
    Returns the BVSS scores for each of the summary pairs and return them in a list. 

    Args:
        - :param: `candidate_summaries` (list of str): candidate summaries - each string is a sumary 
        - :param: `reference_summaries` (list of str): reference summaries - each string is a summary
        - :param: `scoring_approach`    (str): defines whether to use the argmax or mean-based scoring approaches.
        - :param: `model_name`          (str): the specific bert model to use
        - :param: `layer`               (int): the model layer used to retrieve the embedding vectors.
        - :param  'n_gram_encoding'     (int): n-gram encoding level - desginates how many word vectors to combine for each final embedding vector
                                               defaults to none resulting in sentence-level embedding vectors
        - :param: `pool_word_pieces`    (bool): whether or not to preemptively pool word pieces when doing n-gram pooling
                                                only relevant when n_gram_encoding is not None i.e. not for sentence level vectors
        - :param: `language`            (str): the language of the summaries, used for tokenizing the sentences - defaults to english
       
    Return:
        - :param: precision scores (list of float): precision scores for the candidate summaries 
        - :param: recall scores    (list of float): recall scores for the candidate summaries 
        - :param: f1 scores        (list of float): f1 scores for the candidate summaries 
    """
    
    model, tokenizer = get_bert_model(model_name)

    precision_scores = []
    recall_scores = []
    f1_scores = []

    candidate_summaries_sentences = []
    reference_summaries_sentences = []

    # Returns each summary as a list of its sentences
    for i in range(len(candidate_summaries)):
        candidate_summaries_sentences.append(nltk.sent_tokenize(candidate_summaries[i], language= language))
        reference_summaries_sentences.append(nltk.sent_tokenize(reference_summaries[i], language= language))
    

    for i in range(len(candidate_summaries_sentences)):
        
        """
        print('Candidate')
        print(candidate_summaries_sentences[i])

        print('Reference')
        print(reference_summaries_sentences[i])
        """

        candidate_embeddings, reference_embeddings = get_embedding_vectors(candidate_summaries_sentences[i], 
                                                                           reference_summaries_sentences[i], 
                                                                           pool_word_pieces,
                                                                           n_gram_encoding, 
                                                                           layer,
                                                                           model_name,
                                                                           model,
                                                                           tokenizer)


        p, r, f1 = get_bvss(candidate_embeddings, reference_embeddings, scoring_approach)

        precision_scores.append(p)
        recall_scores.append(r)
        f1_scores.append(f1)

    return precision_scores, recall_scores, f1_scores