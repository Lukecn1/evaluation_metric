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

    splitted_words_indicies = set()

    if length_words < length_tokens:
        previous_index = -1
        for i, token in enumerate(tokenized_sentences, 0):
            if token.startswith('##'):
                splitted_words_indicies.update(i-1, i) # adds the previous index and current index to the set. 



def get_ngram_embedding_vectors(candidate_vectors, reference_vectors, n_gram_encoding, pooling_strategy):
    """
    Recieves a list of vectors representing word level vectors, and combines the word-level vectors into n-gram vectors.
    
    Args:
        - :param: `candidate_vectors`, `reference_vectors`  (list of lists of floats (or [CLS]/[SEP]/[UNK]) ): embedding vectors for each token in each sentence -> each sentence is represented a matrix 
                                                                                         entire list contains all the matricies for all of the summaries 

        - :param  'n_gram_encoding' (int): n-gram encoding level - desginates how many word-vectors to combine for each final n-gram-embedding-vector                            
        - :param: `encoding_level` (str): designates whether we wan the server to return word-level encodings for n-gram vectors or sentence level vectors
        - :param: `pooling_strategy` (str): the vector combination strategy - used when 'encoding_level' == 'sentence' 

    Return:
        - :param: - combined_candidate_vectors. combined_reference_vectors (list of list of floats): list of matricies of the embedding vectors for candidate and refernece summaries 
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

    """
    notes: 

    different pooling strategies --> specify which to be used



    include list of valid values for vector_level --> n-gram, sentence etc. (Scale up to n = 1 and up so that 1 vector for an entire summary)

    return_tensors: set the value so that it returns the torch sensors

    dont do comparisons on padding tokens --> find the index values for the 

    implement functionality for selecting multiple layers for embedding vector retrieval

    steps:

    3) Extract and combine at the designated level
    4) return the final vectors
    5) ensure that the method that launches the server is placed in a "main" function call b/c of windows' multi-threading issues 

    """

    launch_bert_as_service_server(model, layer, pool_word_pieces, n_gram_encoding, pooling_strategy)
    bert_client = BertClient()

    candidate_embeddings = []
    reference_embeddings = []
    candidate_tokens = []
    reference_tokens = []

    if not pool_word_pieces:
        for i in range(len(candidate_summaries)):
            candidate_embeddings.append(bert_client.encode(candidate_summaries[i]))
            reference_embeddings.append(bert_client.encode(reference_summaries[i]))

    elif pool_word_pieces:
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
    elif n_gram_encoding >= 1:
        get_ngram_embedding_vectors(candidate_embeddings, reference_embeddings, n_gram_encoding, pooling_strategy, ) 