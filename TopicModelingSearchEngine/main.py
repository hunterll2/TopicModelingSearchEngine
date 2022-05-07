import pickle
import pandas as pd
import numpy as np
import spacy
nlp = spacy.load("en_core_web_sm")

from sklearn.metrics.pairwise import cosine_similarity
from gensim.utils import simple_tokenize

import lda
import tf_idf
import word2vec

import preprocess as p

# Search methods
def search_topic(docs, lda_model, dictionary, result, numRes):

    result['bow'] = result['cleaned'].apply(lambda x: list(simple_tokenize(x)))
    result['bow'] = result['bow'].apply(lambda x: dictionary.doc2bow(x))

    # Create empty list with the same size of topics
    result_topics = [None] * result.shape[0]
    for i in range(len(result_topics)):
        result_topics[i] = np.zeros(lda_model.num_topics)

    # 
    for i in range(result.shape[0]):
        doc_bow = result['bow'][i]
        doc_topics = lda_model.get_document_topics(doc_bow)

        for j in range(len(doc_topics)):
            topic_num = doc_topics[j][0]
            topic_probability = doc_topics[j][1]
            result_topics[i][topic_num] = topic_probability

    # Sum all topic probability for each doc
    query_topics = result_topics[0]
    for i in range(1, len(result_topics)):
        query_topics = query_topics + result_topics[i]

    # Divid topic probability by doc num
    for i in range(len(query_topics)):
        if (query_topics[i] != 0):
            query_topics[i] = query_topics[i] / len(result_topics)

    #
    docs['sim'] = docs['topics'].apply(lambda x: cosine_similarity(np.array(query_topics).reshape(1, -1), np.array(x).reshape(1, -1)).item())
    docs.sort_values(by='sim', ascending=False, inplace=True)

    return docs.head(numRes).reset_index(drop=True)

#
def print_result(df, top_n = 10):
    for i in range(0, top_n):
        sim = df['sim'][i]
        title = df['title'][i]
        body = df['body'][i]

        print('\nDoc {} ({}% Similarity)'.format(i+1, round(sim, 2)))
        print('Title: {}\nContent: {}'.format(title, body))

#
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

def search():
    # Load requried data
    corpus = pickle.load(open("dataset/cleaned_corpus_table", 'rb'))
    
    # Search
    method = input("Search method (tf/Topic)>")

    while True:
        query = p.clean(input("Search query>"))

        if (method == "tf"):
            X = pickle.load(open("dataset/X", 'rb'))
            tfidf_vectorizor = pickle.load(open("dataset/tfidf_vectorizor", 'rb'))
            
            result = tf_idf.search(corpus.copy(), X, tfidf_vectorizor, query, 10)

        else:
            w2v_model = word2vec.Word2Vec.load("dataset/w2v_model")
            lda_model = lda.LdaModel.load("dataset/lda_model")
            dictionary = lda.Dictionary.load("dataset/cleaned_corpus_dictionary")

            word2vec_result = word2vec.search(corpus, w2v_model, query)

            result = search_topic(corpus, lda_model, dictionary, word2vec_result, 10)

        # Done
        print_result(result)

# Start the program
trainOrSearch = input("Train or Search? (t/S)>")

if trainOrSearch.lower() == "t":
    train()

else:
    search()