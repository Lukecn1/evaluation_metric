import pandas as pd
import encode
import BVSS
import utils
from nltk import sent_tokenize

df = pd.read_csv('C:/Users/Lukas/ITU/Master_Thesis/Metric_paper/Data/TAC/TAC_preprocessed.csv', sep= '\t')

output_path = 'C:/Users/Lukas/ITU/Master_Thesis/Metric_paper/Results/TAC_summaries_scores/'

def generate_sentence_level_scores(df):

    cand_sums = df['Summary']

    ref1 = df['refsum1']
    ref2 = df['refsum2']
    ref3 = df['refsum3']
    ref4 = df['refsum4']

    # Parameters
    model_name = 'deepset/sentence_bert'
    n_gram = None
    wp = False
    scorings = ['argmax', 'mean']
    layers = [6,7,8,9,10,11,12]

    model, tokenizer = utils.get_bert_model(model_name)

    for scoring in scorings:
        for layer in layers:

            iteration_name = 'sentence-bert' + '_' + scoring +'_' + str(layer)

            print('Currently scoring: ', iteration_name, '...')

            final_precision_scores = []
            final_recall_scores = []
            final_f1_scores = []

            for i, _ in enumerate(cand_sums):

                cand_sum = sent_tokenize(cand_sums[i], language= 'english')
                ref1_sum = sent_tokenize(ref1[i], language= 'english')
                ref2_sum = sent_tokenize(ref2[i], language= 'english')
                ref3_sum = sent_tokenize(ref3[i], language= 'english')
                ref4_sum = sent_tokenize(ref4[i], language= 'english')

                cand_vecs, cand_tokens = encode.get_embeddings(cand_sum, model, layer, tokenizer)
                ref1_vecs, ref1_tokens = encode.get_embeddings(ref1_sum, model, layer, tokenizer) 
                ref2_vecs, ref2_tokens = encode.get_embeddings(ref2_sum, model, layer, tokenizer)
                ref3_vecs, ref3_tokens = encode.get_embeddings(ref3_sum, model, layer, tokenizer)
                ref4_vecs, ref4_tokens = encode.get_embeddings(ref4_sum, model, layer, tokenizer)

                final_cand_vecs = encode.get_ngram_embedding_vectors(cand_vecs, n_gram, wp, cand_tokens)
                final_ref1_vecs = encode.get_ngram_embedding_vectors(ref1_vecs, n_gram, wp, ref1_tokens)
                final_ref2_vecs = encode.get_ngram_embedding_vectors(ref2_vecs, n_gram, wp, ref2_tokens)
                final_ref3_vecs = encode.get_ngram_embedding_vectors(ref3_vecs, n_gram, wp, ref3_tokens)
                final_ref4_vecs = encode.get_ngram_embedding_vectors(ref4_vecs, n_gram, wp, ref4_tokens)

                p_1, r_1, f1_1 = BVSS.get_bvss(final_cand_vecs, final_ref1_vecs, scoring)
                p_2, r_2, f1_2 = BVSS.get_bvss(final_cand_vecs, final_ref2_vecs, scoring)
                p_3, r_3, f1_3 = BVSS.get_bvss(final_cand_vecs, final_ref3_vecs, scoring)
                p_4, r_4, f1_4 = BVSS.get_bvss(final_cand_vecs, final_ref4_vecs, scoring)

                final_precision_scores.append((p_1 + p_2 + p_3 + p_4) / 4)
                final_recall_scores.append((r_1 + r_2 + r_3 + r_4) / 4) 
                final_f1_scores.append((f1_1 + f1_2 + f1_3 + f1_4) / 4) 

            iteration_df = pd.DataFrame()
            iteration_df['precision'] = final_precision_scores
            iteration_df['recall'] = final_recall_scores
            iteration_df['f1'] = final_f1_scores

            iteration_df.to_csv(output_path + iteration_name + '.csv', sep= '\t')
            print('Finished scoring iteration: ', iteration_name)

#generate_sentence_level_scores(df)


def generate_n_gram_scores(df):

    cand_sums = df['Summary']
    ref1 = df['refsum1']
    ref2 = df['refsum2']
    ref3 = df['refsum3']
    ref4 = df['refsum4']

    # Parameters
    models = ['bert-base-uncased', 'bert-base-cased', 'deepset/sentence_bert']
    scorings = ['argmax', 'mean']
    n_grams = [2,3] 
    wps = [True, False]
    layers = [6,7,8,9,10,11,12]

    for model_name in models:

        model, tokenizer = utils.get_bert_model(model_name)

        for layer in layers:

            for i, _ in enumerate(cand_sums):
                cand_sum = sent_tokenize(cand_sums[i], language= 'english')
                ref1_sum = sent_tokenize(ref1[i], language= 'english')
                ref2_sum = sent_tokenize(ref2[i], language= 'english')
                ref3_sum = sent_tokenize(ref3[i], language= 'english')
                ref4_sum = sent_tokenize(ref4[i], language= 'english')

                cand_vecs, cand_tokens = encode.get_embeddings(cand_sum, model, layer, tokenizer)
                ref1_vecs, ref1_tokens = encode.get_embeddings(ref1_sum, model, layer, tokenizer) 
                ref2_vecs, ref2_tokens = encode.get_embeddings(ref2_sum, model, layer, tokenizer)
                ref3_vecs, ref3_tokens = encode.get_embeddings(ref3_sum, model, layer, tokenizer)
                ref4_vecs, ref4_tokens = encode.get_embeddings(ref4_sum, model, layer, tokenizer)
                print('Encoding completed')


            for scoring in scorings:
                for ngram in n_grams:
                    for wp in wps:
                        
                        iteration_name = ""

                        if model == 'deepset/sentence_bert':
                            iteration_name = 'deepset-sentence-bert' + '_' + scoring + '_' + str(wp) + '_' + str(ngram) + '_' + str(layer)
                        else:
                            iteration_name = model_name + '_' + scoring + '_' + str(wp) + '_' + str(ngram) + '_' + str(layer)

                        print('Currently scoring: ', iteration_name, '...')

                        final_precision_scores = []
                        final_recall_scores = []
                        final_f1_scores = []

                        for i, _ in enumerate(cand_sums):

                            final_cand_vecs = encode.get_ngram_embedding_vectors(cand_vecs, ngram, wp, cand_tokens)
                            final_ref1_vecs = encode.get_ngram_embedding_vectors(ref1_vecs, ngram, wp, ref1_tokens)
                            final_ref2_vecs = encode.get_ngram_embedding_vectors(ref2_vecs, ngram, wp, ref2_tokens)
                            final_ref3_vecs = encode.get_ngram_embedding_vectors(ref3_vecs, ngram, wp, ref3_tokens)
                            final_ref4_vecs = encode.get_ngram_embedding_vectors(ref4_vecs, ngram, wp, ref4_tokens)

                            p_1, r_1, f1_1 = BVSS.get_bvss(final_cand_vecs, final_ref1_vecs, scoring)
                            p_2, r_2, f1_2 = BVSS.get_bvss(final_cand_vecs, final_ref2_vecs, scoring)
                            p_3, r_3, f1_3 = BVSS.get_bvss(final_cand_vecs, final_ref3_vecs, scoring)
                            p_4, r_4, f1_4 = BVSS.get_bvss(final_cand_vecs, final_ref4_vecs, scoring)

                            final_precision_scores.append((p_1 + p_2 + p_3 + p_4) / 4)
                            final_recall_scores.append((r_1 + r_2 + r_3 + r_4) / 4) 
                            final_f1_scores.append((f1_1 + f1_2 + f1_3 + f1_4) / 4) 

                        iteration_df = pd.DataFrame()
                        iteration_df['precision'] = final_precision_scores
                        iteration_df['recall'] = final_recall_scores
                        iteration_df['f1'] = final_f1_scores

                        iteration_df.to_csv(output_path + iteration_name + '.csv', sep= '\t')
                        print('Finished scoring iteration: ', iteration_name)

generate_n_gram_scores(df[0:5])