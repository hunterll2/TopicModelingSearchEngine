import pickle
import time
import pandas as pd

import constants
import preprocess

from tf_idf import search as search_tfidf
from word2vec import search as search_word2vec
from _main import searching as search_topic

from tf_idf import show_similar_documents

import lda
import preprocess as p

corpus = pickle.load(open("dataset/"+constants.CLEANED_CORPUS_TABLE, "rb"))
corpus = corpus.dropna()
corpus['vector'] = pickle.load(open("dataset/"+constants.W2V_MODEL+"_vectors", "rb"))
lda_model = lda.LdaModel.load("dataset/"+constants.LDA_MODEL)

#queries = pd.read_table("dataset/queries.tsv")
#best_queries = pickle.load(open("dataset/best_queries", 'rb'))

#qs = [   
#"What diseases affect your bones",
#"What percentage of American children live in poverty ?",
#"How to Write an Introduction for Your Paper",
#"Global Warming",
#"both dna and rna are polymers that are made up of",
#"bill gate",
#"fico",  
#"1 cubic yard to square feet",
#"what compound has both ionic and covalent bonds",
#"what are the charges on protons, electrons and neutrons",
#]

#numRes = 10

#print("\n========================================================\n")

#c = 1
#for q in qs:
#    qc = p.clean(q)
#    print("Query {}: {} ({})".format(c, q, qc))
#    res1_1, res1_2 = search_tfidf(qc, 10)
#    show_similar_documents(corpus, res1_1, res1_2)
#    c += 1

#print("\n========================================================\n")

#c = 1
#for q in qs:
#    qc = p.clean(q)
#    print("Query {}: {} ({})".format(c, q, qc))
#    res1_1, res1_2 = search_tfidf(q, 10)
#    show_similar_documents(corpus, res1_1, res1_2)
#    c += 1



#print("\n========================================================\n")
#for i in range(0, numRes):
#    print("* Doc "+str(i+1)+" ("+str(res2['similarity'][i])+"): "+res2['title'][i])
#    print(res2['body'][i])
#    print()


#print("\n========================================================\n")

#c = 0
#for q in qs:
#    c += 1
#    qc = p.clean(q)
    
#    print("Query {}: {} ({})".format(c, q, qc))
    
#    res2 = search_word2vec(corpus, qc)
#    res3 = search_topic(corpus, lda_model, res2, qc, 10)

#    for i in range(0, 10):
#        print('\nDoc {} ({}% Similarity)'.format(i+1, round(res3['sim'][i], 2)))
#        print('Title: {}\nContent: {}'.format(res3['title'][i], res3['body'][i]))

#print("\n========================================================\n")

#c = 0
#for q in qs:
#    c += 1
#    qc = p.clean(q)
    
#    print("Query {}: {} ({})".format(c, q, qc))
    
#    res2 = search_word2vec(corpus, q)
#    res3 = search_topic(corpus, lda_model, res2, q, 10)

#    for i in range(0, 10):
#        print('\nDoc {} ({}% Similarity)'.format(i+1, round(res3['sim'][i], 2)))
#        print('Title: {}\nContent: {}'.format(res3['title'][i], res3['body'][i]))


q = input(">")
qc = p.clean(q)

method = input("select search method (tf/topic)>")

print("\nQuery: {} ({})".format(q, qc))
    
if (method == "tf"):
    res1_1, res1_2 = search_tfidf(qc, 10)
    show_similar_documents(corpus, res1_1, res1_2)

elif method == "topic":
    res2 = search_word2vec(corpus, qc)
    res3 = search_topic(corpus, lda_model, res2, qc, 10)

    for i in range(0, 10):
        print('\nDoc {} ({}% Similarity)'.format(i+1, round(res3['sim'][i], 2)))
        print('Title: {}\nContent: {}'.format(res3['title'][i], res3['body'][i]))

print()