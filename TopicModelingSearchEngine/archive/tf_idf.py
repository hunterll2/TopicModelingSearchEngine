import constants
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import spacy

nlp = spacy.load("en_core_web_sm")

from sys import getsizeof as size
import time
import preprocess
#

corpus = pickle.load(open("dataset/"+constants.CLEANED_CORPUS_TABLE, "rb"))
corpus = corpus[:50000]
corpus = corpus.dropna()

docs = [" ".join(x) for x in corpus['list']]

vectorizer = TfidfVectorizer(dtype=np.float32)

X = vectorizer.fit_transform(docs)

X = X.T.toarray()

df = pd.DataFrame(X, index=vectorizer.get_feature_names_out())

del X

def get_similar_articles(q, df):
  print("\n\n\n # query:", q,'\n')
  
  # Convert the query become a vector
  q=preprocess.clean(q)
  q = [q]
  q_vec = vectorizer.transform(q).toarray().reshape(df.shape[0],)
  sim = {}
  
  # Calculate the similarity
  for i in range(df.shape[1]):
    sim[i] = np.dot(df.loc[:, i].values, q_vec) / np.linalg.norm(df.loc[:, i]) * np.linalg.norm(q_vec)
  
  # Sort the values 
  sim_sorted = sorted(sim.items(), key=lambda x: x[1], reverse=True)
  
  # Print the articles and their similarity values
  for k, v in sim_sorted:
    if v > 0.40:
      print("Similarity:", v)
      print(corpus['title'][k][:500])
      print(corpus['body'][k][:500])
      print()

#queries = [
#    "what was the immediate impact of the success of the manhattan project?",
#    "elegxo meaning",
#    "what does physical medicine do",
#    "feeding rice cereal how many times per day",
#    "most dependable affordable cars",
#    "lithophile definition",
#    "what is a flail chest",
#    "put yourself on child support in texas",
#    "what happens in a wrist sprain",
#    "what are rhetorical topics"
#    ]


#for q in queries:
#    start = time.time()
#    get_similar_articles(q, df)
#    end = time.time()
#    print('Time: %0.2fs' % (end - start))

while(True):
    q = input("\n>")
    get_similar_articles(q, df)

print()