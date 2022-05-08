import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def train(corpus, max_features=5000, max_df=0.95, min_df=2):
    docs = [x for x in corpus['cleaned']]

    """ Creates a tf-idf matrix for the `corpus` using sklearn.""" 
    tfidf_vectorizor = TfidfVectorizer(decode_error='replace', 
                                       strip_accents='unicode', 
                                       analyzer='word',
                                       ngram_range=(1, 1),
                                       norm='l2', 
                                       use_idf=True, 
                                       smooth_idf=True, 
                                       sublinear_tf=True,
                                       max_df=max_df, 
                                       min_df=min_df)

    X = tfidf_vectorizor.fit_transform(docs)

    pickle.dump(tfidf_vectorizor, open("dataset/tfidf_vectorizor", 'wb'))
    pickle.dump(X, open("dataset/X", 'wb'))

def search(df, X, vectorizor, query, top_k=5):
    """ Vectorizes the `query` via `vectorizor` and calculates the cosine similarity of
    the `query` and `X` (all the documents) and returns the `top_k` similar documents."""

    # Vectorize the query to the same length as documents
    query_vec = vectorizor.transform([query])
    
    # Compute the cosine similarity between query_vec and all the documents
    df["sim"] = cosine_similarity(X, query_vec).flatten()
    
    # Sort the similar documents from the most similar to less similar
    df = df.sort_values(by='sim', ascending=False)

    return df.head(top_k).reset_index(drop=True)