

bert_model_directories = {'bert-base-uncased' : 'C:/Users/Lukas/ITU/Master_Thesis/Transformers/bert/uncased_L-12_H-768_A-12/', 
                          'bert-base-cased' : 'C:/Users/Lukas/ITU/Master_Thesis/Transformers/bert/cased_L-12_H-768_A-12/', 
                          'mbert' : 'C:/Users/Lukas/ITU/Master_Thesis/Transformers/bert/multi_cased_L-12_H-768_A-12/'}

# Convention of layer naming for the bert-as-service server
layers_base = {1 : '-12', 
                 2 : '-11', 
                 3 : '-10', 
                 4 : '-9',
                 5 : '-8',
                 6 : '-7',
                 7 : '-6',
                 8 : '-5', 
                 9 : '-4',
                 10 : '-3',
                 11 : '-2',
                 12 : '-1'}

pooling_strategies = ['REDUCE_MEAN', 
                      'READUCE_MAX', 
                      'REDUCE_MEAN_MAX',
                      'CLS_TOKEN',
                      'SEP_TOKEN']

