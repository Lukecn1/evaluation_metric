import pandas as pd
import seaborn as sbs
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


df_paws_train = pd.read_csv('C:/Users/Lukas/ITU/Master_Thesis/Metric_paper/Data/paws/final/data_paws_train_final.csv', sep = '\t')
df_paws_test = pd.read_csv('C:/Users/Lukas/ITU/Master_Thesis/Metric_paper/Data/paws/final/data_paws_test_final.csv', sep = '\t')

df_paws_all = df_paws_train.append(df_paws_test, ignore_index= True)
paws_non_score_columns = ['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'Unnamed: 0.1.1.1', 'Unnamed: 0.1.1.1.1', 'Unnamed: 0.1.1.1.1.1', 'id', 'sentence1', 'sentence2', 'label', 'rouge_1', 'rouge_2', 'rouge_l']
paws_score_columns = [x for x in df_paws_all.columns.tolist() if x not in paws_non_score_columns]
#print(len(df_paws_all))
"""
"""

"""
df_msrp_train = pd.read_csv('C:/Users/Lukas/ITU/Master_Thesis/Metric_paper/Data/msrp/data_msrp_train_final.csv', sep = '\t', error_bad_lines=False)
df_msrp_test = pd.read_csv('C:/Users/Lukas/ITU/Master_Thesis/Metric_paper/Data/msrp/data_msrp_test_final.csv', sep = '\t', error_bad_lines=False)

df_msrp_all = df_msrp_train.append(df_msrp_test, ignore_index= True)
msrp_non_score_columns = ['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'Unnamed: 0.1.1.1', 'Unnamed: 0.1.1.1.1', 'Unnamed: 0.1.1.1.1.1', 'Unnamed: 0.1.1.1.1.1.1', 'Unnamed: 0.1.1.1.1.1.1.1', 'Unnamed: 0.1.1.1.1.1.1.1.1', 'Unnamed: 0.1.1.1.1.1.1.1.1.1', 'Unnamed: 0.1.1.1.1.1.1.1.1.1.1', 'Unnamed: 0.1.1.1.1.1.1.1.1.1.1.1', 'Unnamed: 0.1.1.1.1.1.1.1.1.1.1.1.1', 'Label', '#1 ID', '#2 ID', 'sentence1', 'sentence2']
msrp_score_columns = [x for x in df_msrp_all.columns.tolist() if x not in msrp_non_score_columns]
#print(msrp_score_columns)
"""


# Prints the correlation measures for each of the metrics in the dataset
def get_point_biserial(df, score_columns):

    for metric in score_columns:
        print(metric, ': ', stats.pointbiserialr(df['Label'], df[metric]))


# Trains an runs a Logistic regression model for classification of label based on the different metrics
# predictor parameter is the name of the metric to be used for predicting semantic similarity
def run_log_reg(df_train, df_test, metric, target_feature, report):

   #x_train, x_test, y_train, y_test = train_test_split(df[metric], df[target_feature], test_size = 0.25, random_state = 5)
    
    x_train = df_train[metric]
    x_test = df_test[metric]
    
    y_train = df_train[target_feature]
    y_test = df_test[target_feature]    

    x_train = x_train.values.reshape(-1, 1)
    x_test = x_test.values.reshape(-1, 1)

    logistic_model = LogisticRegression()
    logistic_model.fit(x_train, y_train)

    predicts = logistic_model.predict(x_test)
    print('CLASSIFICATION RESULTS', metric)
    if report:
        print(metrics.classification_report(y_test, predicts))
    print(metrics.accuracy_score(y_test, predicts))

#get_point_biserial(df_msrp_all, msrp_score_columns)


"""
for metric in paws_score_columns:
    run_log_reg(df_paws_train, df_paws_test, metric, 'label', True)
"""