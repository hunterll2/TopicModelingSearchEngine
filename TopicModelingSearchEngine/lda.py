import numpy as np
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from sklearn.metrics.pairwise import cosine_similarity

def train(corpus):
    print("\n# Start training LDA module.")

    # make list of docs. so that [ ['word1', 'word2',..], ['word1', 'word2',..], .. ]
    docs = list(corpus['list'])

    # Create Dictionary
    dictionary = Dictionary(docs)

    # Create Term-Document Frequency matrix
    docs_as_bow = [dictionary.doc2bow(doc) for doc in docs]

    # Get user config data
    print("\nEnter config data:")
    num_topics = int(input("Number of topics>"))
    passes = int(input("Number of passes>"))
    alpha = float(input("Alpha value (1)>"))
    eta = float(input("Eta value (0.001)>"))

    # Start training the lda model
    lda = LdaModel(corpus=docs_as_bow, id2word=dictionary, num_topics=num_topics, passes=passes, alpha=alpha, eta=eta)
    
    # Use the "LDA model" on the corpus to find the topics for each document
    corpus['bow'] = docs_as_bow

    # Create new column to store for each documents its topics
    corpus['topics'] = None
    corpus['topics'] = corpus['topics'].apply(lambda x: np.zeros(lda.num_topics))

    for i in range(len(corpus)):
            doc_bow = corpus['bow'][i]
            doc_topics = lda.get_document_topics(doc_bow)

            for j in range(len(doc_topics)):
                topic_num = doc_topics[j][0]
                topic_probability = doc_topics[j][1]

                corpus['topics'][i][topic_num] = topic_probability

    # done
    print("\n* LDA module trained.")

    return lda, corpus

def search(corpus, word2vec_result, top_n = 10):
    # 1. Sum all topic probability for each document
    query_topics = word2vec_result['topics'][0]

    for i in range(1, len(word2vec_result)):
        query_topics = query_topics + word2vec_result['topics'][i]

    # 2. Divid topic probability by document number
    for i in range(len(query_topics)):
        if (query_topics[i] != 0):
            query_topics[i] = query_topics[i] / len(word2vec_result)

    # 3. Calculate the similarity between each "document topics" and "query topics"
    corpus['sim'] = corpus['topics'].apply(lambda doc_topics: 
                                   cosine_similarity(np.array(query_topics).reshape(1, -1), np.array(doc_topics).reshape(1, -1)).item())

    # 4. Sort the documents and return the result
    corpus = corpus.sort_values(by='sim', ascending=False)

    return corpus.head(top_n).reset_index(drop=True)