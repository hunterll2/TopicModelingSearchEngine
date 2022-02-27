import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import time
import pickle
from tabulate import tabulate

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import preprocess
import constants

def get_embedding_w2v(w2v_model, doc_tokens):
    embeddings = []

    if len(doc_tokens)<1:
        return np.zeros(300)
    else:
        for tok in doc_tokens:
            if tok in w2v_model.wv.index_to_key:
                embeddings.append(w2v_model.wv.word_vec(tok))
            else:
                embeddings.append(np.random.rand(300))
        
        # mean the vectors of individual words to get the vector of the document
        return np.mean(embeddings, axis=0)

def ranking_ir(corpus, vector):
  # ranking documents
  documents=corpus[['docid','title','body']].copy()
  documents['similarity']=corpus['vector'].apply(lambda x: cosine_similarity(np.array(vector).reshape(1, -1), np.array(x).reshape(1, -1)).item())
  documents.sort_values(by='similarity', ascending=False, inplace=True)
  
  return documents.head(10).reset_index(drop=True)

def train():
    # Get corpus
    print("Select corpus: ")
    print("- cleaned corpus (50,000)")
    corpus_name = input(">")

    corpus = preprocess.get_corpus(corpus_name)

    txts = corpus['cleaned']
    train_data=[]
    for i in txts:
        train_data.append(i.split())

    # traind w2v model
    print("\nStart training word2vec model")

    start = time.time()
    w2v_model = Word2Vec(train_data, vector_size=300, min_count=2, window=5, sg=1, workers=4)
    end = time.time()
    print('Time to train the model: %0.2fs' % (end - start))

    print("\nEnter w2v trained model name: ")
    w2v_name = input(">")
    w2v_model.save(w2v_name)

    # Create corpus vectors
    print("\nStart createing corpus vectors")

    start = time.time()
    corpus_vectros = corpus['vector'] = corpus['cleaned'].apply(lambda x :get_embedding_w2v(x.split()))
    end = time.time()
    print('Time to create vectors: %0.2fs' % (end - start))

    pickle.dump( corpus_vectros, open( w2v_name+"_vectors", "wb" ) )

    # finish
    print("\nThe model has been trained.")
    model_data = {
        "corpus_name":corpus_name,
        "w2v_name":w2v_name,
        "corpus_vectros_name":w2v_name+"_vectors"
    }
    pickle.dump( model_data, open( w2v_name+"_data", "wb" ) )
    return

def search():
    # load trained model
    print("Select trained model: ")
    print("- w2v_model_50_data")
    #saved_model_name = input(">")
    saved_model_name = "w2v_model_50_data"
    model_data = pickle.load(open(saved_model_name, "rb"))

    corpus = pickle.load(open(model_data['corpus_name'], "rb"))
    corpus = corpus.dropna()

    w2v_model = Word2Vec.load(model_data['w2v_name'])
    
    #corpus_vectros = pickle.load(open(model_data['corpus_vectros_name'], "rb"))
    #corpus['vector'] = corpus_vectros

    print("strat createing corpus vectors")
    start = time.time()
    corpus['vector'] = corpus['cleaned'].apply(lambda x :get_embedding_w2v(w2v_model, x.split()))
    pickle.dump( corpus['vector'], open( "corpus_vectros_50", "wb" ) )
    end = time.time()
    print('Time to create vectors: %0.2fs' % (end - start))

    

    print("Enter seaech query: ")
    query = input(">")

    # pre-process Query
    query=preprocess.clean(query)

    # generating vector
    vector=get_embedding_w2v(w2v_model, query.split())

    start = time.time()
    result = ranking_ir(corpus, vector)
    end = time.time()
    print('Time to search: %0.2fs' % (end - start))

    print(tabulate(result, headers = 'keys', tablefmt = 'psql'))

    # show result
    print("done")

search()