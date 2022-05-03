import numpy as np
from gensim.models import Word2Vec
import time
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import constants

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
    # Get corpus
    #corpus = pickle.load(open(constants.CLEANED_CORPUS_TABLE, "rb"))

    #
    txts = corpus['cleaned']
    train_data=[]
    for i in txts:
        train_data.append(i.split())

    # Get user config data
    print("\nEnter config data:")
    vector_size = int(input("vector_size (300)>"))
    min_count = int(input("min_count (2)>"))
    window = float(input("window (5)>"))
    sg = float(input("sg (1)>"))
    workers = float(input("workers (4)>"))

    # traind w2v model
    print("\nStart training word2vec model")

    start = time.time()
    w2v_model = Word2Vec(train_data, vector_size=vector_size, min_count=min_count, window=window, sg=sg, workers=workers)
    end = time.time()
    print('Time to train the model: %0.2fs' % (end - start))

    # Create corpus vectors
    print("\nStart createing corpus vectors")

    start = time.time()
    corpus_vectros = corpus['vector'] = corpus['cleaned'].apply(lambda x :get_embedding_w2v(x.split()))
    end = time.time()
    print('Time to create vectors: %0.2fs' % (end - start))

    # Saving
    w2v_model.save("dataset/"+constants.W2V_MODEL)
    pickle.dump( corpus_vectros, open("dataset/"+constants.W2V_MODEL+"_vectors", "wb" ) )

    #
    print("\nThe model has been trained.")
    return

def search(corpus, w2v_model, query):
    # generating vector
    vector = get_embedding_w2v(w2v_model, query.split())

    # rank documents
    documents=corpus[['docid','title','body', 'cleaned']].copy()
    documents['similarity']=corpus['vector'].apply(lambda x: cosine_similarity(np.array(vector).reshape(1, -1), np.array(x).reshape(1, -1)).item())
    documents.sort_values(by='similarity', ascending=False, inplace=True)
  
    return documents.head(10).reset_index(drop=True)