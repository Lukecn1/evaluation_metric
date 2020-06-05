
"""
Responsible for loading the datasets of sentence pairs and generate features for analysis(embeddings, ROUGE scores, cosine similarities, etc.)

"""
import torch
import tensorflow as tf
import transformers
import bert_score 
import pandas as pd 
import bert_score
from bert_serving.client import BertClient
from matplotlib import pyplot as plt
from sklearn.decomposition import pca
from scipy.spatial.distance import cosine
from ROUGE import rouge_L, rouge_N



################## DATASET LOADING ######################################



# PAWS Datasets
#df_paws_test = pd.read_csv('C:/Users/Lukas/ITU/Master_Thesis/Metric_paper/Data/paws/final/test.tsv', sep = '\t')
#df_paws_train = pd.read_csv('C:/Users/Lukas/ITU/Master_Thesis/Metric_paper/Data/paws/final/train.tsv', sep = '\t')
#df_paws_dev = pd.read_csv('C:/Users/Lukas/ITU/Master_Thesis/Metric_paper/Data/paws/final/dev_version.tsv', sep = '\t') # Used for developing and testing the functionality of the funcitons implemented here

# ADD DATA-LOADER FUNCTIONS FOR MULTILINGUAL PAWS DATASET


# MSRP - Datasets
#df_msrp_test = pd.read_csv('C:/Users/Lukas/ITU/Master_Thesis/Metric_paper/Data/MSRP/msr_paraphrase_test.txt', sep = '\t', error_bad_lines=False)
#df_msrp_train = pd.read_csv('C:/Users/Lukas/ITU/Master_Thesis/Metric_paper/Data/msrp/msr_paraphrase_train.txt', sep = '\t', error_bad_lines=False)

 


#model = BertModel.from_pretrained('bert-base-uncased')
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


###### BERT-AS-SERVICE CLIENT ####### 
# 
# ENTIRE CODE CANNOT RUN IF THIS LINE IS UNCOMMENTED AND A SERVER IS NOT UP AN RUNNING LOCALLY ON YOUR MACHINE
bert_client = BertClient(ip = 'localhost', check_length = False)





######### HELPER FUNCTIONS ##########


# Takes a list of vectors - reduces their dimensionality with pca - returns the list with the reduced vectors
def reduce_vector_dimensionality(vectors):

    # POTENTIAL AVENUE FOR FUTURE RESEARCH - THE EFFECTS OF DIMENSIONALITY REDUCTION ON DOWNSTREAM TASKS
    # What are the sweetspots for different downstream tasks? 
    # What are the potential downsides? 
    # Poor mans BERT paper --> prunes the network prior to embedding, rather than after --> makes it easier (computationally) to train

    # use a map and or lambda function to decrease the code footprint
    # also pythonic way is often faster than doint loops and or other imperative ways
    pass    


# Generates the BERT-Vectors for each sentence-pair
# Needs to have a bert-as-service server setup 
def gen_bert_vectors(df, sent1, sent2):

    # bert-as-service clients handles batching automatically
    vectors_sent1 = bert_client.encode(df[sent1].values.tolist())
    vectors_sent2 = bert_client.encode(df[sent2].values.tolist())
    
    return vectors_sent1, vectors_sent2




###### SCORING FUNCTIONS #######

# Generate the Bert Vector Similarity Score (BVSS) as the cosine similarity of the vector-pairs of each sentence pair
# model_type = string name of the BERT model used (e.g. bert-base-uncased)
# layer = integer denoting the transformer layer the sentence embeddings are extracted from
def gen_bvss(df, vectors_sent1, vectors_sent2, model_type, layer):

    cosines = []
    
    print('BVSS ' + model_type + str(layer) + ' STATUS: LAUNCHED')
    for i in range(len(vectors_sent1)):
        cos_sim = 1 - cosine(vectors_sent1[i], vectors_sent2[i])
        cosines.append(cos_sim)

    df['bvss_' + model_type + '_' + str(layer)] = cosines
    print('BVSS ' + model_type + str(layer) + ' STATUS: DONE')

    return df



# Generates r1, r2 and rl scores for each sentence pair 
def gen_rouge_scores(df, sent1, sent2, remove_stopwords):

    print('ROUGE-1 STATUS: LAUNCHED')
    df['rouge1_stopwords:' + str(remove_stopwords)] = df.apply(lambda row: rouge_N(row[sent1], row[sent2], 1, remove_stopwords), axis = 1)
    print('ROUGE-1 STATUS: DONE')
    print('ROUGE-2 STATUS: LAUNCHED')
    df['rouge2_stopwords:' + str(remove_stopwords)] = df.apply(lambda row: rouge_N(row[sent1], row[sent2], 2, remove_stopwords), axis = 1)
    print('ROUGE-2 STATUS: DONE')
    print('ROUGE-L STATUS: LAUNCHED')
    df['rougel_stopwords:' + str(remove_stopwords)] = df.apply(lambda row: rouge_L(row[sent1], row[sent2], remove_stopwords), axis = 1)
    print('ROUGE-L STATUS: DONE', '\n')
    
    return df



# Calcualtes the BERT_score for the sentence pair, from the paper https://arxiv.org/pdf/1904.09675.pdf
# Returns a list of f1 scores for the sentence pairs given
# 
# model_type parameter designates the model to be used
# layer paramewter denotes the layer to retrieve the sentence embeddings from
def gen_bert_score(df, sent1, sent2, model_type, layer):

    p, r, f1 = bert_score.score(df[sent1].values.tolist(), df[sent2].values.tolist(), model_type = model_type, num_layers = layer, lang = 'en', batch_size = 256)

    return f1.tolist()




######## CONTROL FLOW SECTION ######### 
#SETUP AND RUNNING THE FUNCTIONS FOR GENERATING THE METRICS FOR THE DATA


# Computes the metrics and returns the dataframe  
# sent1, sent2 is the names of the sentence features
def compute_metrics(df, sent1, sent2, model_type, layer, rouge, remove_stopwords):
    
    print('RETRIEVING VECTORS FROM MODEL: ' + model_type + ' LAYER: ' + str(layer))
    vects1, vects2 = gen_bert_vectors(df, sent1, sent2)
    print('VECTORS RETRIEVED')

    df_bert = gen_bvss(df, vects1, vects2, model_type, layer)

    if rouge:
        df_rouge = gen_rouge_scores(df_bert, sent1, sent2, remove_stopwords)
        return df_rouge
    else:
        return df_bert



# Control flow function that generates all the BERTscore metrics for the paraphrase detection analysis
# Iterates through the different bert-models and the different layers to be used for the analysis
# sent1 and sent2 are the names of the sentence features in the dataset
def gen_bertscore_metrics(df, sent1, sent2):

    base_layers = [i for i in range(3, 13)]
    large_layers = [i for i in range(3, 25)]

    bert_models = ['bert-base-uncased', 'bert-base-cased-finetuned-mrpc']

    for model in bert_models:
        if 'base' in model:
            for layer in base_layers:
                metric_iteration = 'bertscore_' + model + '_' + str(layer) 
                print('BEGINNING ' + model + ' ' + str(layer))
                bertscores = gen_bert_score(df, sent1, sent2, model, layer)
                df[metric_iteration] = bertscores
                print('ITERATION ' + metric_iteration + ' DONE')
        else:
            for layer in large_layers:
                metric_iteration = 'bertscore_' + model + '_' + str(layer)
                print('BEGINNING ' + metric_iteration)
                bertscores = gen_bert_score(df, sent1, sent2, model, layer)
                df[metric_iteration] = bertscores
                print('ITERATION ' + metric_iteration + ' DONE')
    return df




# GENERATE THE DATAFRAME WITH THE METRICS 
#"df_msrp_test =  pd.read_csv('C:/Users/Lukas/ITU/Master_Thesis/Metric_paper/code/data_msrp_test_v1_4.csv', sep = '\t', error_bad_lines=False)
"""
df_paws_train = pd.read_csv('C:/Users/Lukas/ITU/Master_Thesis/Metric_paper/code/data_paws_train_final.csv', sep = '\t', error_bad_lines=False)

df_processed = gen_rouge_scores(df_paws_train, 'sentence1', 'sentence2', True)

df_processed.to_csv('Data_paws_train_final.csv', sep='\t', encoding='utf-8')
"""