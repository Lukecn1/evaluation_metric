import torch
import transformers
import nltk
import numpy as np
from bert_serving.client import BertClient

from utils import launch_bert_as_service_server, terminate_server



def pool_vectors(vectors):
    """
    Takes the average of the n vectors and returns the single result vector
    
    Args:
        - :param: 'vectors' (list of vectors): the embedding vectors to be combined
    Return:
        - :param: 'result_vector' (list of floats): the single output vector resulting from the combination
    """
    result_vector = np.mean(np.array(vectors), axis= 0)

    return result_vector.tolist()


def get_valid_range(tokens):
    """
    Finds the index of the last token before the [SEP] token,
    denoting the index of the lat word embedding vector 
    
    Args:
        - :param: 'tokens' (list of strings): the BERT tokens from a given sentence
    Return:
        - :param: valid_range (int): index of the last valid word vector, i.e. the index of the token before [SEP]
    """

    valid_range = -1
    
    for i, token in enumerate(tokens, 0):
        if token == '[SEP]':
            valid_range = i - 1

    return valid_range
    


# SCOPE CURRENTLY: 1 SENTENCE -> LIST OF VECTORS
def combine_word_piece_vectors(embedding_vectors, tokens):
    """
    Identifies the words that have been split by the BERT wordpiece tokenizer,
    and pools together their individual vectors into 1. 
    
    Args:
        - :param: 'embedding_vectors' ()
        - :param: 'tokens' ()
    Return:
        - :param: 'pooled_wordpiece_vectors' (list of lists of floats): embeddings vectors post pooling of word-pieces
        - :param: 'valid_range' (int): index of the last vector in the matrix - used for the get_ngram_embedding_vectors function
    """

    pooled_wordpiece_vectors = []
    valid_range = 0
    j = 0

    for i, token in enumerate(tokens, 0):
        if token.startswith('##'):
            pool_vectors([embedding_vectors[i], pooled_wordpiece_vectors[j-1]])
        else:
            pooled_wordpiece_vectors[j] = embedding_vectors[i]
            j += 1
    
    valid_range = len(pooled_wordpiece_vectors) - 1

    return pooled_wordpiece_vectors, valid_range


# SCOPE CURRENTLY: 1 SUMMARY -> LIST OF LIST VECTORS
def get_ngram_embedding_vectors(embedding_vectors, n_gram_encoding):
    """
    Recieves a list of vectors representing word level vectors, and combines the word-level vectors into n-gram vectors.
    
    Args:
        - :param: `embedding_vectors` (list of lists of floats): embedding vectors for each token in each sentence of the summaries 
        - :param  'n_gram_encoding'   (int): n-gram encoding level - desginates how many word-vectors to combine for each final n-gram-embedding-vector                            
        
    Return:
        - :param: - combined_embedding_vectors (list of list of floats): list of matricies of the embedding vectors for the summaries 
    """

    """
    check whiteboard strategy

    should return 1 list of vectors for each summary! -> this makes the computation of the score quite a lot easier to manage 
        - Each time a vector is derived -> put it into a list for that summary
        - once the summary has no more vectors to be combined -> place that final summary "matrix" in a list
        - once all summaries have been completed -> return the final list of all summaries
    """



def get_embedding_vectors(candidate_summaries, reference_summaries, model, layer, pool_word_pieces, n_gram_encoding = None):
    """
    Generates the embedding vectors for the given sentences/tokens
    Uses the BERT-as-Service Client to produce the vectors. 

    Args:
        - :param: `candidate_summaries` (list of list of strings): candidate summaries to be encoded - each summary should be represented as a list of sentences
        - :param: `reference_summaries` (list of list of strings): reference summaries to be encoded - each summary should be represented as a list of sentences
        - :param: `model` (str): the specific bert model to use
        - :param: `layer` (int): the layer of representation to use.
        - :param: `pool_word_pieces` (bool): if True, it checks each token and checks  
        - :param  'n_gram_encoding' (int): n-gram encoding level - desginates how many word vectors to combine for each final embedding vector
                                           if 'None' -> embedding level defaults to the sentence level of each individual sentence
      
    Return:
        - :param: candidate_embeddings, (list of lists of float): list of embedding vectors for the candidate summaries
        - :param: reference_embeddings, (list of lists of float): list of embedding vectors for the reference summaries
    """

    bert_client = BertClient()

    candidate_embeddings = []
    reference_embeddings = []

    candidate_tokens = []
    reference_tokens = []

    if n_gram_encoding == None:
        for i in range(len(candidate_summaries)):
            candidate_embeddings.append(bert_client.encode(candidate_summaries[i]))
            reference_embeddings.append(bert_client.encode(reference_summaries[i]))

    elif n_gram_encoding >= 1:
        for i in range(len(candidate_summaries)):
            cand_embeddings, cand_tokens = bert_client.encode(candidate_summaries[i], show_tokens = True)
            ref_embeddings, ref_tokens = bert_client.encode(reference_summaries[i], show_tokens = True)
            
            candidate_embeddings.append(cand_embeddings)
            candidate_tokens.append(cand_tokens)
            
            reference_embeddings.append(ref_embeddings)
            reference_tokens.append(ref_tokens)


    if n_gram_encoding == None:
        return candidate_embeddings, reference_embeddings

    # RECODE THE BELOW CODE TO FOLLOW THE UPDATED CONTROL FLOW

    elif n_gram_encoding >= 1 and not pool_word_pieces:
        candidate_embeddings = get_ngram_embedding_vectors(candidate_embeddings, n_gram_encoding) 
        reference_embeddings = get_ngram_embedding_vectors(reference_embeddings, n_gram_encoding)

    elif n_gram_encoding >= 1 and pool_word_pieces:
        cand_pooled_wordpieces = combine_word_piece_vectors(candidate_embeddings, candidate_tokens, candidate_summaries)
        ref_pooled_wordpieces = combine_word_piece_vectors(reference_embeddings, reference_tokens, reference_summaries)

        candidate_embeddings = get_ngram_embedding_vectors(cand_pooled_wordpieces, n_gram_encoding)
        reference_embeddings = get_ngram_embedding_vectors(ref_pooled_wordpieces, n_gram_encoding)

    return candidate_embeddings, reference_embeddings
