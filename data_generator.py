import pandas as pd
from time import process_time
from BVSS import get_bvss_scores, get_bertscore

df_tac = pd.read_csv('C:/Users/Lukas/ITU/Master_Thesis/Metric_paper/Data/TAC/TAC_ROUGE-HUMAN-REF.csv', sep= '\t')


#print(cands)

"""
print(df_tac['refsum2'])
print(df_tac['refsum3'])
print(df_tac['refsum4'])
"""

"""
for ref in refs:
    print(ref)
"""

def preprocess(df):

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

        end = len(s6)
        if not s6[end - 2] == '.':
            s6 = s6 + '.'

        final_summaries.append(s6)

    return  final_summaries


df_proc = preprocess(df_tac)

cands = df_proc['Summary'].tolist()
refs = df_proc['refsum1'].tolist()

new_refs = preprocess_references(refs)
new_cands = preprocess_candidates(cands)

"""
for i, row in df_proc.iterrows():
    if isinstance(row['Summary'], float):
        print(i)

"""

"""
for i, cand in enumerate(cands, 0):
    if isinstance(cand, float):
        print(i)
"""


"""
print(new_cands[49])
print()
print()
print(new_refs[49])
"""

t_start = process_time()
print(get_bvss_scores(new_cands[850:1000], new_refs[850:1000], 'argmax', 'bert-base-uncased', 11, 2, pool_word_pieces= True, language= 'english'))
t_end = process_time()

print('TIME FOR RUN: ',  t_end-t_start)
"""
"""