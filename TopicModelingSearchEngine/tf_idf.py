from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

def create_tfidf_features(corpus, max_features=5000, max_df=0.95, min_df=2):
    """ Creates a tf-idf matrix for the `corpus` using sklearn. stop_words='english' max_features=max_features""" 

    tfidf_vectorizor = TfidfVectorizer(decode_error='replace', strip_accents='unicode', analyzer='word',
                                       ngram_range=(1, 1),
                                       norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True,
                                       max_df=max_df, min_df=min_df)

    X = tfidf_vectorizor.fit_transform(corpus)

    return X, tfidf_vectorizor

def train(corpus):
    docs = [x for x in corpus['cleaned']]

    X, tfidf_vectorizor = create_tfidf_features(docs)

    pickle.dump(X, open("dataset/X", 'wb'))
    pickle.dump(tfidf_vectorizor, open("dataset/tfidf_vectorizor", 'wb'))
    return

def search(corpus, X, vectorizor, query, top_k=5):
    """ Vectorizes the `query` via `vectorizor` and calculates the cosine similarity of
    the `query` and `X` (all the documents) and returns the `top_k` similar documents."""

    # Vectorize the query to the same length as documents
    query_vec = vectorizor.transform([query])
    
    # Compute the cosine similarity between query_vec and all the documents
    cosine_similarities = cosine_similarity(X, query_vec).flatten()
    
    # Sort the similar documents from the most similar to less similar and return the indices
    most_similar_doc_indices = np.argsort(cosine_similarities, axis=0)[:-top_k-1:-1]

    #
    corpus["sim"] = [0] * len(corpus)
    for index in most_similar_doc_indices:
        i = corpus.index[index]
        corpus["sim"][i] = cosine_similarities[index]

    corpus.sort_values(by='sim', ascending=False, inplace=True)

    return corpus.head(top_k).reset_index(drop=True)