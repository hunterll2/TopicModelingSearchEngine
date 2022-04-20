import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import constants

#from sklearn.feature_extraction.text import TfidfVectorizer, _preprocess
#from sklearn.metrics.pairwise import _return_float_dtype, cosine_similarity

__all__ = ['preprocess', 'create_tfidf_features', 'calculate_similarity', 'show_similar_documents', 'run_query_loop']

# Cell
def __preprocess(title, body=None):
    """ Preprocess the input, i.e. lowercase, remove html tags, special character and digits."""
    text = ''
    if body is None:
        text = title
    else:
        text = title + body
    # to lower case
    text = text.lower()

    # remove tags
    text = re.sub("</?.*?>"," <> ", text)

    # remove special characters and digits
    text = re.sub("(\\d|\\W)+"," ", text).strip()
    return text

def create_tfidf_features(corpus, max_features=5000, max_df=0.95, min_df=2):
    """ Creates a tf-idf matrix for the `corpus` using sklearn. stop_words='english' max_features=max_features""" 

    tfidf_vectorizor = TfidfVectorizer(decode_error='replace', strip_accents='unicode', analyzer='word',
                                       ngram_range=(1, 1),
                                       norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True,
                                       max_df=max_df, min_df=min_df)

    X = tfidf_vectorizor.fit_transform(corpus)

    return X, tfidf_vectorizor

def calculate_similarity(X, vectorizor, query, top_k=5):
    """ Vectorizes the `query` via `vectorizor` and calculates the cosine similarity of
    the `query` and `X` (all the documents) and returns the `top_k` similar documents."""

    # Vectorize the query to the same length as documents
    query_vec = vectorizor.transform([query])
    
    # Compute the cosine similarity between query_vec and all the documents
    cosine_similarities = cosine_similarity(X,query_vec).flatten()
    
    # Sort the similar documents from the most similar to less similar and return the indices
    most_similar_doc_indices = np.argsort(cosine_similarities, axis=0)[:-top_k-1:-1]
    
    return (most_similar_doc_indices, cosine_similarities)

def show_similar_documents(df, cosine_similarities, similar_doc_indices):
    """ Prints the most similar documents using indices in the `similar_doc_indices` vector."""
    
    counter = 1
    
    for index in similar_doc_indices:
        print('\nDoc {} ({}% Similarity)'.format(counter, round(cosine_similarities[index], 2)))

        i = df.index[index]

        print('Title: {}\nContent: {}'.format(df['title'][i], df['body'][i]))

        counter += 1

#
def run_query_loop():
    """ Asks user to enter a query to search."""
    while True:
        try:
            handle_query()
        except KeyboardInterrupt:
            break
    return

#
#corpus = pickle.load(open("dataset/"+constants.CLEANED_CORPUS_TABLE, "rb"))
X = pickle.load(open("dataset/X", 'rb'))
tfidf_vectorizor = pickle.load(open("dataset/tfidf_vectorizor", 'rb'))

def train():
    docs = [x for x in corpus['cleaned']]

    X, tfidf_vectorizor = create_tfidf_features(docs)

    pickle.dump(X, open("dataset/X", 'wb'))
    pickle.dump(tfidf_vectorizor, open("dataset/tfidf_vectorizor", 'wb'))

def search(q, numRes):
    sim_vecs, cosine_similarities = calculate_similarity(X, tfidf_vectorizor, q, numRes)
    return cosine_similarities, sim_vecs

#run_query_loop()