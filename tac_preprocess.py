import pandas as pd
import nltk
import re




def remove_empty_summaries(df):
    for i, row in df.iterrows():
        if isinstance(row['Summary'], float):
            df = df.drop(i)
    return df 


def preprocess_references(summaries):
    """
    Returns a list of strings - each string is a summary
    """
    final_summaries = []

    for summary in summaries:

        s1 = summary.replace('[', '')
        s2 = s1.replace(']', '')
        s3 = s2.replace("\\n", "")
        s4 = s3.replace('\\', "") # Removes single backslash
        s5 = s4.replace('",', '')
        s6 = s5.replace("',", '')
        s7 = s6.replace('"', '')
        s8 = s7.replace("'", '')
            
        final_summaries.append(s8)

    return final_summaries


def preprocess_candidates(summaries):
    """
    Returns a list of strings - each string is a summary
    """
    final_summaries = []

    for summary in summaries:

        s1 = summary.replace("\n", " ")
        s2 = s1.replace('\\', "") # Removes single backslash
        s3 = s2.replace('",', '')
        s4 = s3.replace("',", '')
        s5 = s4.replace('"', '')
        s6 = s5.replace("'", '')
        s6 = s6.replace('.', '. ')

        end = len(s6)
        if not s6[end - 2] == '.':
            s6 = s6 + '.'

        s6 = s6.replace('.  .', '.')

        final_summaries.append(s6)

    return  final_summaries

def preproces(df):
    df = remove_empty_summaries(df)

    df['Summary'] = preprocess_candidates(df['Summary'])

    reference_summaries = ['refsum1','refsum2', 'refsum3', 'refsum4']
    for ref_sum in reference_summaries:
        df[ref_sum] = preprocess_references(df[ref_sum])

    return df


df_tac = pd.read_csv('C:/Users/Lukas/ITU/Master_Thesis/Metric_paper/Data/TAC/TAC_ROUGE-HUMAN-REF_1.csv', sep= '\t')
df_processed = preproces(df_tac)
out_path = 'C:/Users/Lukas/ITU/Master_Thesis/Metric_paper/Data/TAC/TAC_preprocessed.csv'
df_processed.to_csv(out_path, sep = '\t', encoding= 'utf-8')