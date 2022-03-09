from word2vec import search
import lda
import constants
import spacy
import pickle
from gensim.corpora import Dictionary
nlp = spacy.load("en_core_web_sm")
from gensim.utils import simple_tokenize
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

dictionary = Dictionary.load("dataset/"+constants.CLEANED_CORPUS_DICTIONARY)
lda_model = lda.LdaModel.load("dataset/"+constants.LDA_MODEL)
docs = pickle.load(open("dataset/"+constants.CLEANED_CORPUS_TABLE, "rb"))

##


docs['bow'] = docs['cleaned'].apply(lambda x: list(simple_tokenize(x)))
docs['bow'] = docs['bow'].apply(lambda x: dictionary.doc2bow(x))

dt = [None] * docs.shape[0]
for i in range(len(dt)):
    dt[i] = np.zeros(lda_model.num_topics)

for i in range(docs.shape[0]):
    d = docs['bow'][i]
    d = lda_model.get_document_topics(d)
    for j in range(len(d)):
        t = d[j][0]
        p = d[j][1]
        dt[i][t] = p

##
result = search("product")

result['bow'] = result['cleaned'].apply(lambda x: list(simple_tokenize(x)))
result['bow'] = result['bow'].apply(lambda x: dictionary.doc2bow(x))

# Create empty list with the same size of topics
rt = [None] * result.shape[0]
for i in range(len(rt)):
    rt[i] = np.zeros(lda_model.num_topics)

# 
for i in range(result.shape[0]):
    d = result['bow'][i]
    d = lda_model.get_document_topics(d)
    for j in range(len(d)):
        t = d[j][0]
        p = d[j][1]
        rt[i][t] = p

# Sum all topic probability for each doc
qt = rt[0]
for i in range(1, len(rt)):
    qt = qt + rt[i]

# Divid topic probability by doc num
for i in range(len(qt)):
    if (qt[i] != 0):
        qt[i] = qt[i] / len(rt)

##
result = []
for i in range(len(dt)):
    doc = dt[i]
    doc_sim = cosine_similarity(np.array(qt).reshape(1, -1), np.array(doc).reshape(1, -1)).item()
    result.append(doc_sim)

for i in range(len(result)):
    print("doc "+str(i)+" = "+str(doc_sim))

print()