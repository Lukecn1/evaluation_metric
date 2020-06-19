# Text Similarity Assesment using BERT

Using the contextually-dependent embeddings obtained from BERT to assess the similarity of a candidate- and reference-text using cosine similarity. 
The main application of this metric is to assess the similarity between a candidate- and reference-summary.

### Algorithm
The scoring algorithm consists of two central steps:

#### 1) Encoding 
Obtaining embedding vectors from a pretrained BERT-based model. We use the BERT as service package (https://github.com/hanxiao/bert-as-service) to retrieve the embedding vectors.

#### 2) Scoring 
Calculating the score using cosine similarity. 



### Installation requirements
* Python version >= 3.6
* Tensorflow version =< 1.15.2 (1.15.2 recommended for stability reasons)
* bert-as-service (https://github.com/hanxiao/bert-as-service)
* nltk (packages for tokenization at the least)
