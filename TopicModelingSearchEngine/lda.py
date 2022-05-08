import pickle
import numpy as np
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from sklearn.metrics.pairwise import cosine_similarity

def train(df):
    # make list of docs
    articles = [x for x in df['list']]

    # Create Dictionary
    dictionary = Dictionary(articles)

    # Create Term-Document Frequency matrix
    docs_as_bow = [dictionary.doc2bow(article) for article in articles]

    # Get user config data
    print("\nEnter config data:")
    num_topics = int(input("Number of topics>"))
    passes = int(input("Number of passes>"))
    alpha = float(input("Alpha value (1)>"))
    eta = float(input("Eta value (0.001)>"))

    # Start training the lda model
    print("\nStart training the LDA model")
    lda = LdaModel(corpus=docs_as_bow, id2word=dictionary, num_topics=num_topics, passes=passes, alpha=alpha, eta=eta)

    # Saving
    lda.save("dataset/lda_model")

    # Use the "LDA model" on the corpus to find the topics for each document
    df['bow'] = docs_as_bow

    df['topics'] = None
    df['topics'] = df['topics'].apply(lambda x: np.zeros(lda.num_topics))

    for i in range(len(df)):
            doc_bow = df['bow'][i]
            doc_topics = lda.get_document_topics(doc_bow)

            for j in range(len(doc_topics)):
                topic_num = doc_topics[j][0]
                topic_probability = doc_topics[j][1]

                df['topics'][i][topic_num] = topic_probability

    pickle.dump( df, open( "dataset/cleaned_corpus_table", "wb" ) )

def search(df, top_docs):
    num_result = len(top_docs)

    # 1. Sum all topic probability for each document
    query_topics = top_docs['topics'][0]

    for i in range(1, num_result):
        query_topics = query_topics + top_docs['topics'][i]

    # 2. Divid topic probability by document number
    for i in range(num_result):
        if (query_topics[i] != 0):
            query_topics[i] = query_topics[i] / num_result

    # 3. Calculate the similarity between each "document topics" and "query topics"
    df['sim'] = df['topics'].apply(lambda x: 
                                   cosine_similarity(np.array(query_topics).reshape(1, -1), np.array(x).reshape(1, -1)).item())

    # 4. Sort the documents and return only top N
    df = df.sort_values(by='sim', ascending=False)

    return df.head(num_result).reset_index(drop=True)