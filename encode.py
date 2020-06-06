import torch
import transformers
import nltk
from bert_serving.client import BertClient

from utils import launch_bert_as_service_server, terminate_server



def combine_word_piece_vectors():
    """
    Identifies the words that have been split by the BERT wordpiece tokenizer,
    and pools together their individual vectors into 1. 
    
    Args:
        - :param: 
    Return:
        - :param: 
    """
    """
    Only return the vectors that remain, i.e. only return the word vectors and ignore the special tokens and the vectors for the padding 
    """


    splitted_words_indicies = set()

    if length_words < length_tokens:
        previous_index = -1
        for i, token in enumerate(tokenized_sentences, 0):
            if token.startswith('##'):
                splitted_words_indicies.update(i-1, i) # adds the previous index and current index to the set. 



def get_ngram_embedding_vectors(embedding_vectors, n_gram_encoding, pooling_strategy, range):
    """
    Recieves a list of vectors representing word level vectors, and combines the word-level vectors into n-gram vectors.
    
    Args:
        - :param: `embedding_vectors` (list of lists of floats): embedding vectors for each token in each sentence of the summaries 
        - :param  'n_gram_encoding'   (int): n-gram encoding level - desginates how many word-vectors to combine for each final n-gram-embedding-vector                            
        - :param: `encoding_level`    (str): designates whether we wan the server to return word-level encodings for n-gram vectors or sentence level vectors
        - :param: `pooling_strategy`  (str): the vector combination strategy - used when 'encoding_level' == 'sentence' 
        - :param: `pooling_strategy`  (str): defines the index for the last word vectors 
        

    Return:
        - :param: - combined_embedding_vectors (list of list of floats): list of matricies of the embedding vectors for the summaries 
    """

    """
    - Make a slicing functionality that makes the n-gram slices for each sentence matrix. 
        1) Check number of n-grams between the [CLS] and [SEP] tokens. 
                1.2.1) splitted words and their vectors --> what do we do with these? 
        2) Construct slicing operation that takes the n-grams from these 
            2.1) Finding the correct indicies and then create pooling method to handle
                2.1.1) Pooling method should check that n <= length of the sentence
                2.1.1) Pooling method should ensure that all tokens/words in the sentence gets used

    """
    


def get_embedding_vectors(candidate_summaries, reference_summaries, n_gram_encoding: None, model, layer, pooling_strategy, pool_word_pieces):
    """
    Generates the embedding vectors for the given sentences/tokens
    Uses the BERT as Service Client to produce the vectors. 

    Args:
        - :param: `candidate_summaries` (list of list of strings): candidate summaries to be encoded - each summary should be represented as a list of sentences
        - :param: `reference_summaries` (list of list of strings): reference summaries to be encoded - each summary should be represented as a list of sentences

        - :param  'n_gram_encoding' (int): n-gram encoding level - desginates how many word vectors to combine for each final embedding vector
                                        if 'None' -> embedding level defaults to the sentence level of each individual sentence
        
        - :param: `model` (str): the specific bert model to use
        - :param: `layer` (int): the layer of representation to use.
        - :param: `pooling_strategy` (str): the vector combination strategy 
        - :param: `pool_word_pieces` (bool): if True, it checks each token and checks  

        
    Return:
        - :param: candidate_embeddings, reference_embeddings (list of lists of float): list of embedding vectors for the summaries
                  each summary has a list of vectors (i.e. a matrix)
    """

    launch_bert_as_service_server(model, layer, pool_word_pieces, n_gram_encoding, pooling_strategy)
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

    terminate_server()

    if n_gram_encoding == None:
        return candidate_embeddings, reference_embeddings

    elif n_gram_encoding >= 1 and not pool_word_pieces:
        candidate_embeddings = get_ngram_embedding_vectors(candidate_embeddings, n_gram_encoding, pooling_strategy) 
        reference_embeddings = get_ngram_embedding_vectors(reference_embeddings, n_gram_encoding, pooling_strategy) 

    elif n_gram_encoding >= 1 and pool_word_pieces:
        cand_pooled_wordpieces = combine_word_piece_vectors(candidate_embeddings, candidate_tokens, candidate_summaries)
        ref_pooled_wordpieces = combine_word_piece_vectors(reference_embeddings, reference_tokens, reference_summaries)

        candidate_embeddings = get_ngram_embedding_vectors(cand_pooled_wordpieces, n_gram_encoding, pooling_strategy)
        reference_embeddings = get_ngram_embedding_vectors(ref_pooled_wordpieces, n_gram_encoding, pooling_strategy)

    return candidate_embeddings, reference_embeddings
