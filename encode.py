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
        - :param: valid_range (int): index of the last valid word vector, i.e. the index of the last word
    """
    valid_range = -1
    
    for i, token in enumerate(tokens, 0):
        if token == '[SEP]':
            valid_range = i - 2 # -2 as we dont want the embedding vector for the '.' either

    return valid_range
    


def combine_word_piece_vectors(embedding_vectors, tokens):
    """
    Identifies the words that have been split by the BERT wordpiece tokenizer,
    and pools together their individual vectors into 1. 
    
    Args:
        - :param: 'embedding_vectors' (list of lists of floats): Word-embedding vectors for a single sentence
        - :param: 'tokens'            (list of str): list of the tokens, each corresponding to a embedding vector
    Return:
        - :param: 'pooled_wordpiece_vectors' (list of lists of floats): embeddings vectors post pooling of word-pieces
        - :param: 'valid_range' (int): index of the last vector in the matrix - used for the get_ngram_embedding_vectors function
    """
    pooled_wordpiece_vectors = [i for i in range(len(tokens))]
    valid_range = 0
    j = 0
    poolings = 0

    for i, token in enumerate(tokens, 0):

        if token.startswith('##'):
            pooled_wordpiece_vectors[j-1] = pool_vectors([embedding_vectors[i], pooled_wordpiece_vectors[j-1]])
            poolings += 1
        else:
            pooled_wordpiece_vectors[j] = embedding_vectors[i]
            j += 1
        
        valid_range = i - 2 - poolings # - 2 because we do not want the embedding vector for the '.' either 

    return pooled_wordpiece_vectors[:valid_range + 1], valid_range



def get_ngram_embedding_vectors(embedding_vectors, n_gram_encoding, pool_word_pieces, tokens):
    """
    Recieves a list of vectors representing word level vectors, and combines the word-level vectors into n-gram vectors.
    
    Args:
        - :param: `embedding_vectors` (list of list of lists of floats): embedding vectors for each token in each sentence in a summary -> Each sentence is represented as its own matrix
        - :param  'n_gram_encoding'   (int): n-gram encoding level - desginates how many word-vectors to combine for each final n-gram-embedding-vector                            
        - :param: `pool_word_pieces`  (bool): if True, pools together word-vectors for those words split by the wordpiece tokenizer
        - :param: `tokens`  (list of list of str): the individual tokens for each sentence - used for finding the valid range of vectors
    Return:
        - :param: - final_embeddings (list of list of floats): list of matricies of the embedding vectors for the summaries 
    """
    final_embeddings = []

    for i, sentence_matrix in enumerate(embedding_vectors, 0):
        valid_token_index = get_valid_range(tokens[i])
        
        if pool_word_pieces:
            sentence_matrix, valid_token_index = combine_word_piece_vectors(sentence_matrix, tokens[i])

        if n_gram_encoding > (valid_token_index - 1): # Fewer tokens than desired number of poolings -> Defaults to making a single vector for the sentence
            final_embeddings.append(pool_vectors(sentence_matrix[1:(valid_token_index+1)]))
            continue

        n = 1 # Starting at position 1 to not include the [CLS] token 
        while n + n_gram_encoding <= valid_token_index + 1:
            end_index = n+n_gram_encoding
            final_embeddings.append(pool_vectors(sentence_matrix[n:end_index]))
            n += 1

    return final_embeddings        




def get_embedding_vectors(candidate_summaries, reference_summaries, pool_word_pieces, n_gram_encoding = None):
    """
    Generates the embedding vectors for the given sentences/tokens
    Uses the BERT-as-Service Client to produce the vectors. 

    Args:
        - :param: `candidate_summaries` (list of list of strings): candidate summaries to be encoded - each summary should be represented as a list of sentences
        - :param: `reference_summaries` (list of list of strings): reference summaries to be encoded - each summary should be represented as a list of sentences
        - :param: `pool_word_pieces` (bool): if True, pools together word-vectors for those words split by the wordpiece tokenizer 
        - :param  'n_gram_encoding' (int): n-gram encoding level - desginates how many word vectors to combine for each final embedding vector
                                        if 'None' -> embedding level defaults to the sentence level of each individual sentence
    
    Return:
        - :param: candidate_embeddings, (list of lists of float): list of embedding vectors for the candidate summaries
        - :param: reference_embeddings, (list of lists of float): list of embedding vectors for the reference summaries
    """

    bert_client = BertClient(ip='localhost')

    candidate_embeddings = []
    reference_embeddings = []

    if n_gram_encoding == None:
        for i in range(len(candidate_summaries)):
            candidate_embeddings.append(bert_client.encode(candidate_summaries[i]))
            reference_embeddings.append(bert_client.encode(reference_summaries[i]))

    elif n_gram_encoding >= 1:
        for i in range(len(candidate_summaries)):
            cand_embeddings, cand_tokens = bert_client.encode(candidate_summaries[i], show_tokens = True)
            ref_embeddings, ref_tokens = bert_client.encode(reference_summaries[i], show_tokens = True)

            candidate_embeddings.append(get_ngram_embedding_vectors(cand_embeddings, n_gram_encoding, pool_word_pieces, cand_tokens))
            reference_embeddings.append(get_ngram_embedding_vectors(ref_embeddings, n_gram_encoding, pool_word_pieces, ref_tokens))

    return candidate_embeddings, reference_embeddings