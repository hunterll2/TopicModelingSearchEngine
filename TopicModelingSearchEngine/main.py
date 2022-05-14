import pickle
import pandas as pd

import lda
import tf_idf
import word2vec

import preprocess as p

# Helpers
def save(obj, name):
    pickle.dump( obj, open( "dataset/" + name, "wb" ) )

# Train methods
def preprocess():
    corpus = pd.read_table('dataset/sample_corpus.tsv', encoding="latin1")
    
    corpus_size = int(input("Number of docs>"))
    
    new_corpus = p.create_new(corpus, corpus_size)

    save(new_corpus, "cleaned_corpus_table")

def train():
    # load the saved corpus
    corpus = pickle.load(open("dataset/cleaned_corpus_table", 'rb'))

    # train the requested module
    module = input("Module to train [ TFIDF (TF) / LDA / Word2Vec (w2v) ] >")

    if module == "tf":
        tfidf_vectorizor, docs_vectors = tf_idf.train(corpus)

        save(tfidf_vectorizor, "tfidf_vectorizor")
        save(docs_vectors, "docs_vectors")
    
    elif module == "lda":
        lda_model, edited_corpus = lda.train(corpus)

        lda_model.save("dataset/lda_model")
        save(edited_corpus, "cleaned_corpus_table")
    
    elif module == "w2v":
        w2v_model, edited_corpus = word2vec.train(corpus)

        w2v_model.save("dataset/w2v_model")
        save(edited_corpus, "cleaned_corpus_table")

# Seach Methods
def search():
    # Load requried data
    corpus = pickle.load(open("dataset/cleaned_corpus_table", 'rb'))
    
    # Search
    method = input("Search method (tf/Topic)>")

    while True:
        query = p.clean(input("\nSearch query>"))

        if query == "": break

        if (method == "tf"):
            docs_vectors = pickle.load(open("dataset/docs_vectors", 'rb'))
            tfidf_vectorizor = pickle.load(open("dataset/tfidf_vectorizor", 'rb'))
            
            result = tf_idf.search(corpus.copy(), docs_vectors, tfidf_vectorizor, query)
            
        else:
            w2v_model = word2vec.Word2Vec.load("dataset/w2v_model")

            word2vec_result = word2vec.search(corpus.copy(), w2v_model, query)

            result = lda.search(corpus.copy(), word2vec_result)

        # Done
        print_result(result)
          
# Print Method
def print_result(result):
    for i, doc in result.iterrows():
        print('\nDoc {} ({}% Similarity)'.format(i+1, round( doc['sim'] * 100 )))
        print('Title: {}\nContent: {}'.format(doc['title'], doc['body'][:2000]))

# Start the program
while True:
    operation = input("\nPreprocess (P) / Train (T) / Search (S) >")

    if operation.lower() == "p":
        preprocess()

    elif operation.lower() == "t":
        train()

    else:
        search()