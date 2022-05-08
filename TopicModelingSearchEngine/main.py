import pickle
import pandas as pd

import lda
import tf_idf
import word2vec

import preprocess as p

# Train methods
def train():
    ## 1. Get user options

    newOrOldCorpus = input("Create new corpus? (y/N)>")
    
    # use a new corpus or an old one
    if newOrOldCorpus.lower() == "y":
        df = pd.read_table('./dataset/sample_corpus.tsv', header=None, encoding="latin1")

        numOfDocs = int(input("Number of docs>"))
        
        p.create_new(df, numOfDocs)
        
    # load the saved corpus
    corpus = pickle.load(open("dataset/cleaned_corpus_table", 'rb'))

    ## 2. train the requested module

    moduleToTrain = input("Module to train (tfidf/topic/w2v)>")

    if moduleToTrain == "tf":
        tf_idf.train(corpus)
    
    elif moduleToTrain == "topic":
        lda.train(corpus)
    
    elif moduleToTrain == "w2v":
        word2vec.train(corpus)

    ## 3. done
    print("Module trained.")

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
        print('Title: {}\nContent: {}'.format(doc['title'], doc['body']))

# Start the program
trainOrSearch = input("Train or Search? (t/S)>")

if trainOrSearch.lower() == "t":
    train()

else:
    search()