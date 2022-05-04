import pickle
import pandas as pd
import spacy
nlp = spacy.load("en_core_web_sm")
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from gensim.utils import simple_tokenize
from gensim.corpora import Dictionary
from gensim.models import Word2Vec

import lda

import constants
import preprocess as p

from tf_idf import train as train_tf_idf
from tf_idf import search as search_tfidf

from lda import train as train_topic

from word2vec import train as train_word2vec
from word2vec import search as search_word2vec

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
        print('\nDoc {} ({}% Similarity)'.format(i+1, round(df['sim'][i], 2)))
        print('Title: {}\nContent: {}'.format(df['title'][i], df['body'][i]))
    return

#
def train():
    df = pd.read_table('./dataset/'+constants.ROW_CORPUS_NAME, header=None, encoding="latin1")

    #
    newOrOldCorpus = input("Create new corpus? (y/N)>")

    if (newOrOldCorpus.lower() == "y"):
        numOfDocs = int(input("Number of docs>"))
        
        p.create_new(df, numOfDocs)
        
        corpus = pickle.load(open("dataset/"+constants.CLEANED_CORPUS_TABLE, 'rb'))
    else:
        corpus = pickle.load(open("dataset/"+constants.CLEANED_CORPUS_TABLE, 'rb'))

    #
    moduleToTrain = input("Module to train (tfidf/topic/w2v)>")

    if moduleToTrain == "tf":
        train_tf_idf(corpus)
    elif moduleToTrain == "topic":
        train_topic(corpus)
    elif moduleToTrain == "w2v":
        train_word2vec(corpus)

    print("Module trained.")

    return

def search():
    corpus = pickle.load(open("dataset/"+constants.CLEANED_CORPUS_TABLE, 'rb'))

    method = input("Search method (tf/Topic)>")
    
    # Search
    if (method == "tf"):
        X = pickle.load(open("dataset/X", 'rb'))
        tfidf_vectorizor = pickle.load(open("dataset/tfidf_vectorizor", 'rb'))

        while True:
            query = input("\nSearch query>")
            
            if query == "": break

            query = p.clean(query)

            tfidf_result = search_tfidf(corpus, X, tfidf_vectorizor, query, 10)

            print_result(tfidf_result)

    else:
        w2v_model = Word2Vec.load("dataset/"+constants.W2V_MODEL)
        lda_model = lda.LdaModel.load("dataset/"+constants.LDA_MODEL)
        dictionary = Dictionary.load("dataset/"+constants.CLEANED_CORPUS_DICTIONARY)

        while True:
            query = input("\nSearch query>")
            
            if query == "": break

            query = p.clean(query)
            
            word2vec_result = search_word2vec(corpus, w2v_model, query)
            topic_result = search_topic(corpus, lda_model, dictionary, word2vec_result, 10)

            print_result(topic_result)

    return

#
trainOrSearch = input("Train or Search? (t/S)>")

if (trainOrSearch == 't'):
    train()
else:
    search()