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
    df = pd.read_table('dataset/sample_corpus.tsv', header=None, encoding="latin1")
    
    corpus_size = int(input("Number of docs>"))
    
    new_corpus = p.create_new(df, corpus_size)

    save(new_corpus, "cleaned_corpus_table")

def train():
    # load the saved corpus
    corpus = pickle.load(open("dataset/cleaned_corpus_table", 'rb'))

    # train the requested module
    module = input("Module to train [ TFIDF (TF) / LDA / Word2Vec (w2v) / ALL ] >")

    if module == "tf" or module == "all":
        tfidf_vectorizor, X = tf_idf.train(corpus)

        save(tfidf_vectorizor, "tfidf_vectorizor")
        save(X, "X")
    
    if module == "topic" or module == "all":
        lda_model, edited_corpus = lda.train(corpus)

        lda_model.save("dataset/lda_model")
        save(edited_corpus, "cleaned_corpus_table")
    
    if module == "w2v" or module == "all":
        w2v_model, edited_corpus = word2vec.train(corpus)

        w2v_model.save("dataset/w2v_model")
        save(edited_corpus, "cleaned_corpus_table")

    ## 3. done
    print("\nModule\s trained.")

# Seach Methods
def search():
    # Load requried data
    corpus = pickle.load(open("dataset/cleaned_corpus_table", 'rb'))
    
    # Search
    method = input("Search method (tf/Topic)>")

    while True:
        query = p.clean(input("\nSearch query>"))

        if (method == "tf"):
            X = pickle.load(open("dataset/X", 'rb'))
            tfidf_vectorizor = pickle.load(open("dataset/tfidf_vectorizor", 'rb'))
            
            result = tf_idf.search(corpus.copy(), X, tfidf_vectorizor, query)
            
        else:
            w2v_model = word2vec.Word2Vec.load("dataset/w2v_model")

            word2vec_result = word2vec.search(corpus.copy(), w2v_model, query)

            result = lda.search(corpus.copy(), word2vec_result)

        # Done
        print_result(result)

# Print Method
def print_result(result):
    for i, doc in result.iterrows():
        print('\nDoc {} ({}% Similarity)'.format(i+1, round(doc['sim'] * 100)))
        print('Title: {}\nContent: {}'.format(doc['title'], doc['body'][:2000]))

# Start the program
operation = input("Preprocess (P) / Train (T) / Search (S) >")

if operation.lower() == "p":
    preprocess()

elif operation.lower() == "t":
    train()

else:
    search()