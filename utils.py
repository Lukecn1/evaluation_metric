from bert_serving.client import BertClient
from bert_serving.server.helper import get_args_parser, get_shutdown_parser
from bert_serving.server import BertServer

bert_model_directories = {'bert-base-uncased' : 'C:/Users/Lukas/ITU/Master_Thesis/Transformers/bert/uncased_L-12_H-768_A-12/', 
                          'bert-base-cased' : 'C:/Users/Lukas/ITU/Master_Thesis/Transformers/bert/cased_L-12_H-768_A-12/', 
                          'mbert' : 'C:/Users/Lukas/ITU/Master_Thesis/Transformers/bert/multi_cased_L-12_H-768_A-12/'}

# Convention of layer naming for the bert-as-service server
layers_bert_base = {1 : '-12', 
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

def launch_bert_as_service_server(model_name, layer, return_tokens, encoding_level = None, pooling_strategy = None):
    """
    Launches a BERT-as-service server used to encode the sentences using the designated BERT model
    https://github.com/hanxiao/bert-as-service

    Args:
        - :param: `model_name` (str): the specific bert model to use
        - :param: `layer` (int): the layer of representation to use
        - :param: `return_tokens` (bool): whether or not to return the tokens as well as the embeddings 
        - :param  'encoding_level' (int): n-gram encoding level - desginates how many word vectors to combine for each final embedding vector
                                        if 'none' -> embedding level defaults to the sentence level of each individual sentence
        - :param: `pooling_strategy` (str): the vector combination strategy - used when 'encoding_level' == 'sentence' 
    """
    
    retrieve_tokens = ''
    pooling = 'NONE'

    model_path = bert_model_directories[model_name]
    pooling_layer = layers_bert_base[layer]

    server_parameters = ""
        
    if encoding_level == None:

        if pooling_strategy not in pooling_strategies:
            print('"pooling_strategy" must be defined as one of the following:', pooling_strategies)
            return

        server_parameters = get_args_parser().parse_args(['-model_dir', model_path,
                                        '-port', '5555',
                                        '-port_out', '5556',
                                        '-max_seq_len', '50',                                        
                                        '-pooling_layer', pooling_layer,
                                        '-pooling_strategy', pooling_strategy, 
                                        '-num_workers', '=1'])
    
    elif encoding_level >=1 and not return_tokens:
        server_parameters = get_args_parser().parse_args(['-model_dir', model_path,
                                        '-port', '5555',
                                        '-port_out', '5556',
                                        '-max_seq_len', '50',                                        
                                        '-pooling_layer', pooling_layer,
                                        '-pooling_strategy', 'NONE',
                                        '-num_workers', '=1'])
    
    elif encoding_level >=1 and return_tokens:
                server_parameters = get_args_parser().parse_args(['-model_dir', model_path,
                                        '-port', '5555',
                                        '-port_out', '5556',
                                        '-max_seq_len', '50',                                        
                                        '-pooling_layer', pooling_layer,
                                        '-pooling_strategy', 'NONE',
                                        '-show_tokens_to_client',
                                        '-num_workers', '=1'])
    else:
        print('"encoding_level" must be >=1 or None, see README for descriptions')
        return

    server = BertServer(server_parameters)
    print("LAUNCHING SERVER, PLEASE HOLD", '\n')
    server.start()
    print("SERVER RUNNING, BEGGINING ENCODING...")

def terminate_server():
    print("ENCODINGS COMPLETED, TERMINATING SERVER...")
    shutdown = get_shutdown_parser().parse_args(['-ip','localhost','-port','5555','-timeout','5000'])
    BertServer.shutdown(shutdown)