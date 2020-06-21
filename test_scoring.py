import numpy as np
import math
import unittest
import bert_serving.server
from nltk import sent_tokenize
from bert_serving.client import BertClient
from bert_serving.server.helper import get_args_parser, get_shutdown_parser
from bert_serving.server import BertServer
from scipy.spatial.distance import cosine

from BVSS import get_bvss, get_bvss_scores
from utils import launch_bert_as_service_server, terminate_server
from encode import pool_vectors, get_valid_range, combine_word_piece_vectors, get_ngram_embedding_vectors, get_embedding_vectors

"""
The tests are designed to test the following:
    
    1) Mathematical conceptualization of metric holds
        1.1) Correct and consistent calculations - DONE
            1.1.2) precision, recall & f1 scores for each scoring approach - DONE
        1.2) Scoring approaches are correct and distinct - DONE
        1.3) Test sklearn implementation of cosine similarity - DONE
    
    2) Control flow of the entire program
        2.1) Correct number of scores returned (i.e. 1 pr. candidate summary)
        2.2) If summaries are identical, their score is close to 1 (few decimals)

"""


class testScoringFunctions(unittest.TestCase):

    def test_get_bvss(self):

        candidate_vectors = [[3.4, 6.2, 4.5, -5.5], 
                             [4.3, 2.6, 5.4, 5.5], 
                             [2.4, 3.2, 6.5, 8.5]]
        
        reference_vectors = [[2.3, 7.8, 1.5, -1.8], 
                             [9.9, -1.6, 2.4, 5.5], 
                             [6.4, 3.2, 7.8, 5.1], 
                             [1.4, -7.3, 1.5, -1.5]]


        cand_ort_vectors = [[1, 0, 1, 0]]
        ref_ort_vectors  = [[0, 1, 0, 1]]
        cand_eq_vec = [[0, 1, 0, 1]]   

        results_eq = get_bvss(cand_eq_vec, ref_ort_vectors, 'mean')
        results_ort = get_bvss(cand_ort_vectors, ref_ort_vectors, 'mean')
        results_mean_function = get_bvss(candidate_vectors, reference_vectors, 'mean')
        results_argmax_function = get_bvss(candidate_vectors, reference_vectors, 'argmax')
        
        # Calculations of the scores with the candidate and reference vectors manually
        results_argmax_manual = (0.9101563472457149, 0.6034360551295096, 0.7257187005843395) 
        results_mean_manual =  (0.3661780061962483, 0.36617800619624835, 0.36617800619624835)
        
        self.assertAlmostEqual(results_mean_function[0], results_mean_manual[0])
        self.assertAlmostEqual(results_mean_function[1], results_mean_manual[1])
        self.assertAlmostEqual(results_mean_function[2], results_mean_manual[2])
        self.assertAlmostEqual(results_argmax_function[0], results_argmax_manual[0])
        self.assertAlmostEqual(results_argmax_function[1], results_argmax_manual[1])
        self.assertAlmostEqual(results_argmax_function[2], results_argmax_manual[2])
        self.assertAlmostEqual(results_eq[0], 1.0)
        self.assertAlmostEqual(results_eq[1], 1.0)
        self.assertAlmostEqual(results_eq[2], 1.0)
        self.assertEqual(results_ort[0], 0.0)
        self.assertEqual(results_ort[1], 0.0)


    # tests the control-flow of the entire program, by calling the main function that initializes the calculation of scores
    def test_control_flow(self):

        candidate_summaries = ['First candidate summary for testing. Another sentence for testing purposes. The final phrase is written here.', 
                               'Second candidate summary is written here. It only consists of two sentences.', 
                                'The third and final candidate summary is here. It has more than two sentences. Hence the third text sequence.'
                                ]
        
        reference_summaries = [ 'Here is the first sentence of the reference summary. Only two individual sentences for this summary.', 
                                'Start of the second reference. Testing the controlflow of the embedding functions.',
                                'Lastly a single sentence reference summary.'
                                ]


        precision_scores, recall_scores, f1_scores = get_bvss_scores(candidate_summaries, 
                                                                     reference_summaries, 
                                                                     scoring_approach = 'mean', 
                                                                     model = 'bert-base-uncased', 
                                                                     layer= 11, 
                                                                     n_gram_encoding= 2,
                                                                      pool_word_pieces= True, 
                                                                      language= 'english')

        self.assertEqual(len(precision_scores), 3)
        self.assertEqual(len(recall_scores), 3)
        self.assertEqual(len(f1_scores), 3)
        self.assertGreater()