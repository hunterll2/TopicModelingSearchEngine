from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def train(corpus):
    print("\n# Start training TF-IDF module.")

    docs = list(corpus['cleaned'])

    # prepare tfidf_vectorizor object. used later for convert text into vectors.
    tfidf_vectorizor = TfidfVectorizer(decode_error='replace', 
                                       strip_accents='unicode', 
                                       analyzer='word',
                                       ngram_range=(1, 1),
                                       norm='l2', 
                                       use_idf=True, 
                                       smooth_idf=True, 
                                       sublinear_tf=True,
                                       max_df=0.95, 
                                       min_df=2,
                                       max_features=5000)

    # Creates a tf-idf matrix for the `docs` using tfidf_vectorizor.
    docs_vectors = tfidf_vectorizor.fit_transform(docs)

    # done
    print("* TF-IDF module trained.")

    return tfidf_vectorizor, docs_vectors

def search(corpus, docs_vectors, vectorizor, query, top_n = 10):
    """ Vectorizes the `query` via `vectorizor` and calculates the cosine similarity of
    the `query` and `vectors` (all the documents) and returns the `top_n` similar documents."""

    # Vectorize the query to the same length as documents
    query_vector = vectorizor.transform([query])
    
    # Compute the cosine similarity between query_vec and all the documents
    corpus["sim"] = cosine_similarity(docs_vectors, query_vector).flatten()
    
    # Sort the similar documents from the most similar to less similar
    corpus = corpus.sort_values(by='sim', ascending=False)

    # return top N document after reset the index
    return corpus.head(top_n).reset_index(drop=True)