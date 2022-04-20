import constants
from gensim.corpora import Dictionary
from gensim.utils import simple_tokenize
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

dictionary = Dictionary.load("dataset/"+constants.CLEANED_CORPUS_DICTIONARY)

#lda_model = lda.LdaModel.load("dataset/"+constants.LDA_MODEL)
#docs = pickle.load(open("dataset/"+constants.CLEANED_CORPUS_TABLE, "rb"))

##
#docs = docs.iloc[:50000]
#docs['bow'] = docs['cleaned'].apply(lambda x: list(simple_tokenize(x)))
#docs['bow'] = docs['bow'].apply(lambda x: dictionary.doc2bow(x))
##docs['bow'] = docs['bow'].apply(lambda x: lda_model.get_document_topics(x))

#docs['topics'] = [None] * docs.shape[0]
#docs['topics'] = docs['topics'].apply(lambda x: np.zeros(lda_model.num_topics))

#for (i, columnData) in docs['bow'].iteritems():
#    d = lda_model.get_document_topics(columnData)
#    for j in range(len(d)):
#        tn = d[j][0]
#        tp = d[j][1]
#        docs['topics'][i][tn] = tp

#docs = pickle.load(open("dataset/"+constants.CLEANED_CORPUS_TABLE, "rb"))

##
def searching(docs, lda_model, result, q, numRes):

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
    docs['sim'] = docs['topics'].apply(lambda x: cosine_similarity(np.array(qt).reshape(1, -1), np.array(x).reshape(1, -1)).item())
    docs.sort_values(by='sim', ascending=False, inplace=True)
    result2 = docs.head(numRes).reset_index(drop=True)

    return result2

#while(True):
#    q = input(">")
#    res = searching(q, 10)
#    for i in range(0, 10):
#        print("* Doc "+str(i+1)+" ("+str(res['sim'][i])+"): "+res['title'][i])
#        print(res['body'][i])
#        print()
#    return

