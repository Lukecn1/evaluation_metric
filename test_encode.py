import numpy as np
import unittest
import bert_serving.server
from nltk import sent_tokenize
from bert_serving.client import BertClient
from bert_serving.server.helper import get_args_parser, get_shutdown_parser
from bert_serving.server import BertServer
from scipy.spatial.distance import cosine

from encode import pool_vectors, get_valid_range, combine_word_piece_vectors, get_ngram_embedding_vectors, get_embedding_vectors

"""
Unit tests for the encode functions. 

Each function have at least 1 dedicated test, most have a couple. 

The tests are designed to ensure that the following items are done consistently correctly:

    1) Embedding vectors are produced in accordance with the provided sentences
        1.1) Correct number of vectors for both sentence and word level encodings -  DONE
        1.2) That indicies in each function line up with each other - DONE

    2) Function that pool together vectors take the correct vectors as input.
        2.1) Pooling is done correctly, given the specific pooling strategy - DONE
        2.2) Vectors for [CLS], [SEP] and '.' are not utilized in the comparisons. - DONE 
        2.3) That relevant word vectors are not omitted or lost in the process - DONE
        2.4) That vectors for paddings are never passed to the scoring functions - DONE

    3) That the flow of summaries throughout yields the correct results
        3.1) The chain of function call holds for the correct parameters - DONE
        3.2) That the original order of the input summaries is kept post encoding  - DONE

    4) That the above functionality is kept when changing the BERT model
        4.1) mBERT and other languages than english
        4.2) RoBERTa ? --> It is useable for BERT as service? 
        4.3) Danish BERT --> work with bert as service?
"""

class testEncodeFunctions(unittest.TestCase):

    def test_pool_vectors(self, BERT_vectors = None):
        vectors_zero = [[0, 0, -0, -0], [0, 0,-0, -0], [0, 0, -0, -0], [0, 0, -0, -0]]
        vectors_float = [[3.4, 6.2, 4.5, -5.5], [4.3, 2.6, 5.4, 5.5], [2.4, 3.2, 6.5, 8.5], [5.4, -7.2, 2.5, -1.5]]

        self.assertEqual(len(pool_vectors(vectors_float)), 4) # That the dimensionality of the input vectors are kept
        self.assertEqual(sum(pool_vectors(vectors_zero)), 0)  # Vectors with zero values
        self.assertAlmostEqual(pool_vectors(vectors_float)[0], 3.875)
        self.assertAlmostEqual(pool_vectors(vectors_float)[1], 1.2)
        self.assertAlmostEqual(pool_vectors(vectors_float)[2], 4.725)
        self.assertAlmostEqual(pool_vectors(vectors_float)[3], 1.75)

        if BERT_vectors is not None:
            self.assertEqual(len(pool_vectors(BERT_vectors)), 768)


    def test_get_valid_range(self):
        self.assertEqual(get_valid_range(['[CLS]', 'Test', 'sentence', 'for', 'range', '.', '[SEP]']), 4)


    def test_combine_word_piece_vectors(self):
        vectors_no_split = [[3.4, 6.2, 4.5, -5.5], [3.3, 6.2, 4.5, -5.5], [4.3, 2.6, 5.4, 5.5], [5.4, 3.2, 6.5, 8.5], [5.4, -7.2, 2.5, -1.5]]
        tokens_no_split  = ['[CLS]', 'Token', 'ize', 'word', 'piece', '.', '[SEP]']

        vectors_split_2 = [[3.4, 6.2, 4.5, -5.5], [3.3, 6.2, 4.5, -5.5], [4.3, 2.6, 5.4, 5.5], [5.4, 3.2, 6.5, 8.5], [5.4, -7.2, 2.5, -1.5], [3.4, 6.2, 4.5, -5.5], [3.4, 6.2, 4.5, -5.5]]
        tokens_split_2  = ['[CLS]', 'Token', '##ize', 'word', '##piece', '.', '[SEP]']
        result_split_2 = str(combine_word_piece_vectors(vectors_split_2, tokens_split_2)[0])
        
        vectors_split_2_plus = [[3.4, 6.2, 4.5, -5.5], [3.3, 6.2, 4.5, -5.5], [4.3, 2.6, 5.4, 5.5], [5.4, 3.2, 6.5, 8.5], [5.4, -7.2, 2.5, -1.5], [4.4, 2.2, 9.8, 7.4], [5.6, 7.6, 8.0, 2.2], [3.4, 6.2, 4.5, -5.5], [3.4, 6.2, 4.5, -5.5]]
        tokens_split_2_plus  = ['[CLS]', 'Token', '##ize', 'word', '##piece', 'extra', 'words', '.', '[SEP]']
        result_split_2_plus = str(combine_word_piece_vectors(vectors_split_2_plus, tokens_split_2_plus)[0])
    
        self.assertEqual(combine_word_piece_vectors(vectors_split_2, tokens_no_split)[0], vectors_no_split)    
        self.assertEqual(len(combine_word_piece_vectors(vectors_split_2, tokens_split_2)[0]), 3)        
        self.assertEqual(combine_word_piece_vectors(vectors_split_2, tokens_split_2)[1], 2)        
        self.assertEqual(result_split_2, '[[3.4, 6.2, 4.5, -5.5], [3.8, 4.4, 4.95, 0.0], [5.4, -2.0, 4.5, 3.5]]')
        self.assertEqual(result_split_2_plus, '[[3.4, 6.2, 4.5, -5.5], [3.8, 4.4, 4.95, 0.0], [5.4, -2.0, 4.5, 3.5], [4.4, 2.2, 9.8, 7.4], [5.6, 7.6, 8.0, 2.2]]')
        

    def test_get_ngram_embedding_vectors(self):
    
        start(True)
        bc = BertClient(ip='localhost')
        embeddings, tokens = (bc.encode(['Test for wordpiece tokenizer and pad length.', 'Adding an additional sentence.'], show_tokens=True))

        result_vectors_2 = get_ngram_embedding_vectors(embeddings, 2, True, tokens)
        result_vectors_3 = get_ngram_embedding_vectors(embeddings, 3, True, tokens)
        result_vectors_2_no_pool = get_ngram_embedding_vectors(embeddings, 2, False, tokens)
        result_vectors_3_no_pool = get_ngram_embedding_vectors(embeddings, 3, False, tokens)
        results_large_n = get_ngram_embedding_vectors(embeddings, 12, False, tokens)
        pooled_wp_2 = pool_vectors([embeddings[0][2] ,pool_vectors(embeddings[0][3:5])])
        pooled_wp_2_no_wp = pool_vectors(embeddings[0][2:4])
        pooled_wp_3 = pool_vectors( [embeddings[0][1], embeddings[0][2], pool_vectors(embeddings[0][3:5])])
        pooled_wp_3_no_wp = pool_vectors(embeddings[0][1:4])

        terminate()
        
        self.assertEqual(len(result_vectors_2), 9)
        self.assertEqual(len(result_vectors_3), 7)
        self.assertEqual(len(result_vectors_2_no_pool), 11)
        self.assertEqual(len(result_vectors_3_no_pool), 9)
        self.assertEqual(result_vectors_2[0], pool_vectors(embeddings[0][1:3]))
        self.assertEqual(result_vectors_2[1], pooled_wp_2)
        self.assertEqual(result_vectors_2_no_pool[1], pooled_wp_2_no_wp)
        self.assertEqual(result_vectors_3[0], pooled_wp_3)
        self.assertEqual(result_vectors_3_no_pool[0], pooled_wp_3_no_wp)
        self.assertEqual(len(results_large_n), 2)


    def test_get_embedding_vectors(self):

        candidate_summaries = [ ['First candidate summary for testing.', 'Another sentence for testing purposes.', 'The final phrase is written here.'], 
                                ['Second candidate summary is written here.', 'It only consists of two sentences.'], 
                                ['The third and final candidate summary is here.', 'It has more than two sentences.', 'Hence the third text sequence.'] 
                                ]
        
        reference_summaries = [ ['Here is the first sentence of the reference summary.', 'Only two individual sentences for this summary.'], 
                                ['Start of the second reference.', 'Testing the controlflow of the embedding functions.'], 
                                ['Lastly a single sentence reference summary.'] 
                                ]

        start(False)
        candidate_embeddings, reference_embeddings = get_embedding_vectors(candidate_summaries, reference_summaries, False)
        terminate()

        

        start(True)
        candidate_embeddings, reference_embeddings = get_embedding_vectors(candidate_summaries, reference_summaries, True, 2)
        terminate()

        self.assertEqual(len(candidate_embeddings), 3)
        self.assertEqual(len(candidate_embeddings[0]), 13)
        self.assertEqual(len(candidate_embeddings[1]), 10)
        self.assertEqual(len(candidate_embeddings[2]), 16)
        self.assertEqual(len(reference_embeddings), 3)
        self.assertEqual(len(reference_embeddings[0]), 14)
        self.assertEqual(len(reference_embeddings[1]), 10)
        self.assertEqual(len(reference_embeddings[2]), 5)




def start(word_tokens):

    if word_tokens:
        server_parameters = get_args_parser().parse_args(['-model_dir', 'C:/Users/Lukas/ITU/Master_Thesis/Transformers/bert/uncased_L-12_H-768_A-12/',
                                    '-max_seq_len', '50',                  
                                    '-pooling_layer', '-2',
                                    '-pooling_strategy', 'NONE', 
                                    '-show_tokens_to_client',    
                                    '-num_worker=1'])

        bert_server = BertServer(server_parameters)
        print("LAUNCHING SERVER, PLEASE HOLD", '\n')
        bert_server.start()
    elif not word_tokens:
        server_parameters = get_args_parser().parse_args(['-model_dir', 'C:/Users/Lukas/ITU/Master_Thesis/Transformers/bert/uncased_L-12_H-768_A-12/',
                                '-max_seq_len', '50',                  
                                '-pooling_layer', '-2',
                                '-pooling_strategy', 'REDUCE_MEAN',     
                                '-num_worker=1'])

        bert_server = BertServer(server_parameters)
        print("LAUNCHING SERVER, PLEASE HOLD", '\n')
        bert_server.start()
    


def terminate():
    # Shuts down the server
    shutdown = get_shutdown_parser().parse_args(['-ip','localhost','-port','5555','-timeout','5000'])
    BertServer.shutdown(shutdown)