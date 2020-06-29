import torch
from transformers import BertModel, BertTokenizer, AutoTokenizer, AutoModel
from bert_serving.client import BertClient

from utils import bert_models

def get_bert_model(model_name):
    """
    Retrieves the pretrained model and tokenizer from the transformers library. 

    """
    model = None
    tokenizer = None

    if model_name in bert_models:
        tokenizer = BertTokenizer.from_pretrained(model_name)        
        model = BertModel.from_pretrained(model_name, output_hidden_states=True)
    
    elif model_name == 'sentence-bert':
        # Loads the pretrained bert-base-nli-stsb-mean-tokens model from the sentence transformers git: https://github.com/UKPLab/sentence-transformers
        name = 'deepset/sentence_bert' 
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModel.from_pretrained(name)

    elif model_name == 'nordic-bert':
        model_directory = ''
        tokenizer = AutoTokenizer.from_pretrained(model_directory)
        model = AutoModel.from_pretrained(model_directory)
    
    else: 
        print('model must be specified as one of the supported ones. Check readme for more details')
        return

    return model, tokenizer



def get_embeddings(summary, model_name, layer):

    final_tokens = []
    final_embeddings = []

    model, tokenizer = get_bert_model(model_name)

    model_inputs = tokenizer.batch_encode_plus(summary, max_length = 128)

    input_token_ids = model_inputs['input_ids']
    attention_masks = model_inputs['attention_mask']

    model.eval()
    with torch.no_grad():
        for i, _ in enumerate(input_token_ids, 0):
            vectors = []
            
            inputs =  torch.tensor([input_token_ids[i]])
            masks =  torch.tensor([attention_masks[i]])

            if model_name == 'sentence-bert':
                hidden_states = model(inputs, masks)[1]                
                vectors = hidden_states[0].tolist()

            else:
                hidden_states = model(inputs, masks)[2]
                vectors = hidden_states[layer][0].tolist()

            final_embeddings.append(vectors)

    if model_name == 'sentence-bert':
        return final_embeddings

    for sentence in summary:
        final_tokens.append(tokenizer.tokenize(sentence))

    return final_embeddings, final_tokens


#embeddings, tokens = get_embeddings(summary = ['Test summary for ensuring correct encodings.', 'With more than one sentence.'], 
                                    #model_name = 'sentence-bert', 
                                    #layer = 11)
#print(embeddings)

model, tokenizer = get_bert_model('sentence-bert')

model_inputs = tokenizer.batch_encode_plus(['Test summary for ensuring correct encodings.', 'With more than one sentence.'])

input_token_ids = model_inputs['input_ids']
attention_masks = model_inputs['attention_mask']

model.eval()
with torch.no_grad():
        for i, _ in enumerate(input_token_ids, 0):
            inputs =  torch.tensor([input_token_ids[i]])
            masks =  torch.tensor([attention_masks[i]])
            hidden_states = model(inputs, masks)[1] 
            print(len(hidden_states[0].tolist()))


"""
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

model_inputs = tokenizer.encode_plus('Test sequence for encoding.')

model.eval()
with torch.no_grad():

    model_output = model(input_token_ids, attention_masks)

    hidden_states = model_output[2]
    
    layer = 11
    token_id = 1 # 'Test'


    vector_t =  hidden_states[layer][0][token_id]

    t_vector = vector_t.tolist()

#print(tokenizer.encode_plus('Test sequence for encoding.'))

bc = BertClient(ip = 'localhost')

bc_output = bc.encode(['Test sequence for encoding.'])

b_vec = bc_output[0][1]

for i, _ in enumerate(t_vector):
    print(t_vector[i])
    print(b_vec[i])
    print(b_vec[i] == t_vector[i])

"""

#print(b_vec == t_vector)