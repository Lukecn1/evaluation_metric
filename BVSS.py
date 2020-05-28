import pandas as pd 
import numpy as np
import torch
import nltk
import transformers
import bert_score
from bert_serving.client import BertClient
from bert_serving.server.helper import get_args_parser
from bert_serving.server import BertServer
from scipy.spatial.distance import cosine

from utils import bert_model_directories, layers_base, pooling_strategies



#df_tac = pd.read_csv('C:/Users/Lukas/ITU/Master_Thesis/Metric_paper/Data/TAC/TAC_ROUGE-HUMAN-REF.csv', sep= '\t')

# print(df_tac.columns)

###### HELPER FUNCTIONS ####### 

# Container for the server object

def get_ngram_embedding_vectors(candidate_vectors, reference_vectors, n_gram_encoding, pooling_strategy):
    """
    Recieves a list of vectors representing word level vectors, and combines the word-level vectors into n-gram vectors.
    
    Args:
        - :param: `summaries` (list of list of strings): summaries to be encoded -each summary should be represented as a list of sentences
        - :param  'n_gram_encoding' (int): n-gram encoding level - desginates how many word-vectors to combine for each final n-gram-embedding-vector                            
        - :param: `encoding_level` (str): designates whether we wan the server to return word-level encodings for n-gram vectors or sentence level vectors
        - :param: `pooling_strategy` (str): the vector combination strategy - used when 'encoding_level' == 'sentence' 
    """

    


def launch_bert_as_service_server(model_name, layer, encoding_level = None, pooling_strategy = None):
    """
    Launches a BERT-as-service server used to encode the sentences using the designated BERT model
    https://github.com/hanxiao/bert-as-service

    Args:
        - :param: `model_name` (str): the specific bert model to use
        - :param: `layer` (int): the layer of representation to use
        - :param  'encoding_level' (int): n-gram encoding level - desginates how many word vectors to combine for each final embedding vector
                                        if 'none' -> embedding level defaults to the sentence level of each individual sentence
        - :param: `pooling_strategy` (str): the vector combination strategy - used when 'encoding_level' == 'sentence' 
        
    """

    model_path = bert_model_directories[model_name]
    pooling_layer = layers_base[layer]

    server_parameters = ""
        
    if encoding_level == None:

        if pooling_strategy not in pooling_strategies:
            print('"pooling_strategy" must be defined as one of the following:', pooling_strategies)
            return

        server_parameters = get_args_parser().parse_args(['-model_dir', model_path,
                                        '-port', '5555',
                                        '-port_out', '5556',
                                        '-max_seq_len', 'NONE',                                        
                                        '-pooling_layer', pooling_layer,
                                        '-pooling_strategy', pooling_strategy, 
                                        '-num_workers', '=1'])
    
    elif encoding_level >=1:
        server_parameters = get_args_parser().parse_args(['-model_dir', model_path,
                                        '-port', '5555',
                                        '-port_out', '5556',
                                        '-max_seq_len', 'NONE',                                        
                                        '-pooling_layer', pooling_layer,
                                        '-pooling_strategy', 'NONE',
                                        '-num_workers', '=1'])
    else:
        print('"encoding_level" must be >=1 or None, see README for descriptions')
        return

    server = BertServer(server_parameters)
    print("LAUNCHING SERVER, PLEASE HOLD", '\n')
    server.start()
    # Include a check here that ensures that the server is running before printing the below statement
    print("SERVER RUNNING, BEGGINING ENCODING...")
    



def get_embedding_vectors(candidate_summaries, reference_summaries, n_gram_encoding: None, model, layer, pooling_strategy):
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
        
    Return:
        - :param: embedding_vectors (list of lists of float): list of embedding vectors for the summaries
                  each summary has a list of vectors
    """

    """
    notes: 

    start server from distiinct method for that --> parse the arguments as the guide shows

    different pooling strategies --> specify which to be used

    Should not truncate the sentences --> set encode parameter for this to "not truncate"

    include list of valid values for vector_level --> n-gram, sentence etc. (Scale up to n = 1 and up so that 1 vector for an entire summary)

    return_tensors: set the value so that it returns the torch sensors

    steps:

    1) Start the server
        1.1) Pooling strategies
        1.1) Ensure that server is fully launched and ready to recieve requests before the rest of the method is run --> print statements to prompt the user
    2) Produce the vector
    3) Extract and combine at the designated level
    4) return the final vectors

    """
    
    launch_bert_as_service_server()
    bert_client = BertClient()

    candidate_embeddings = []
    reference_embeddings = []

    #Generates the embedding vectors for each summary. If the 
    for i in range(len(candidate_embeddings)):
        candidate_embeddings.append(bert_client.encode(candidate_summaries[i]))
        reference_embeddings.append(bert_client.encode(reference_summaries[i]))

    print("ENCODINGS COMPLETED, TERMINATING SERVER")
    BertServer.shutdown(port=5555)

    if n_gram_encoding == None:
        return candidate_embeddings, reference_embeddings
    elif n_gram_encoding >= 1:
        get_ngram_embedding_vectors(candidate_embeddings, reference_embeddings, n_gram_encoding, pooling_strategy, ) 




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
