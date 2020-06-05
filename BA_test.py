import bert_serving.server
from bert_serving.client import BertClient
from bert_serving.server.helper import get_args_parser, get_shutdown_parser
from bert_serving.server import BertServer
from scipy.spatial.distance import cosine

def start():
    server_parameters = get_args_parser().parse_args(['-model_dir', 'C:/Users/Lukas/ITU/Master_Thesis/Transformers/bert/uncased_L-12_H-768_A-12/',
                                '-max_seq_len', '50',                  
                                '-pooling_layer', '-2',
                                '-pooling_strategy', 'NONE', 
                                '-show_tokens_to_client',    
                                '-num_worker=1'])

    server = BertServer(server_parameters)
    print("LAUNCHING SERVER, PLEASE HOLD", '\n')
    server.start()
    bc = BertClient(ip='localhost')
    encodings, tokens = (bc.encode(['Test for wordpiece tokenizer and pad length.', 'Adding an additional sentence.'], show_tokens=True))
    print(len(tokens[0]))
    print(len(tokens[1]))
    print(encodings)
    print(encodings[0][10])
    print(tokens[0][10])



if __name__ == "__main__":
    start()
    print("ENCODINGS DONE - SHUTTING DOWN the server", '\n') 
    shutdown = get_shutdown_parser().parse_args(['-ip','localhost','-port','5555','-timeout','5000'])
    BertServer.shutdown(shutdown)