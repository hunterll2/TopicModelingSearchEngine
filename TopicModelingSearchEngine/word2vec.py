import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

def get_embedding_w2v(w2v_model, doc_tokens):
    embeddings = []

    if len(doc_tokens)<1:
        return np.zeros(300)
    else:
        for tok in doc_tokens:
            if tok in w2v_model.wv.index_to_key:
                embeddings.append(w2v_model.wv.get_vector(tok))
            else:
                embeddings.append(np.random.rand(300))
        
        # mean the vectors of individual words to get the vector of the document
        return np.mean(embeddings, axis=0)

def train(corpus):
    print("\n# Start training Word2Vec module.")

    # Get documents text
    txts = list(corpus['list'])

    # Get user config data
    print("\nEnter config data:")
    vector_size = int(input("vector_size (300)>"))
    min_count = int(input("min_count (2)>"))
    window = float(input("window (5)>"))
    sg = float(input("sg (1)>"))
    workers = float(input("workers (4)>"))

    # traind w2v model
    w2v_model = Word2Vec(txts, vector_size=vector_size, min_count=min_count, window=window, sg=sg, workers=workers)

    # Create corpus vectors
    corpus['vector'] = corpus['list'].apply(lambda x :get_embedding_w2v(w2v_model, x))

    # Done
    print("\n* Word2Vec module trained.")

    return w2v_model, corpus

def search(corpus, w2v_model, query, top_n = 10):
    # generating vector
    vector = get_embedding_w2v(w2v_model, query.split())

    # rank documents
    corpus['sim'] = corpus['vector'].apply(lambda x: 
                                   cosine_similarity(np.array(vector).reshape(1, -1), np.array(x).reshape(1, -1)).item())
    
    corpus = corpus.sort_values(by='sim', ascending=False)

    return corpus.head(top_n).reset_index(drop=True)