import pandas as pd
import numpy as np
import dask.dataframe as dd
from tabulate import tabulate
import re
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import spacy
nlp = spacy.load('en_core_web_sm',disable=['ner','parser'])
nlp.max_length=5000000

#df = pd.DataFrame(queries)
#print(tabulate(df, headers = 'keys', tablefmt = 'psql'))

#df1=df.drop([0,100000])
#print(df1)
#df1.to_csv("sample_queries.tsv", sep=" ")



## Queries

queries_train=pd.read_table('./dataset/queries.doctrain.tsv',header=None)
queries_train.columns=['qid','query']
queries_train.head()

queries=queries_train.sample(n=100,random_state=42).reset_index(drop=True)
queries.head()

# Creating Training Set of Queries
training_queries=queries.iloc[:50]
training_queries.head()

## Top Documents

train_top100=pd.read_table('./dataset/msmarco-doctrain-top100',delimiter=' ',header=None)
train_top100.columns=['qid','Q0','docid','rank','score','runstring']
train_top100.head()
print(train_top100)

# Reducing train_top100 for training
training_ranked100=train_top100[train_top100['qid'].isin(training_queries['qid'].unique())].reset_index(drop=True)
training_ranked100.head()

# Labelling Top 10 as 1 and last 10 as 0
rel=list(range(1,11))
nonrel=list(range(91,101))
training_ranked100['rel']=training_ranked100['rank'].apply(lambda x: 1 if x in rel else ( 0 if x in nonrel else np.nan))

# Result set for Training
training_result=training_ranked100.dropna()
training_result['rel']=training_result['rel'].astype(int)
training_result.head()

## corpus 

df=pd.read_table('./dataset/sample_corpus.tsv',header=None, encoding="latin1")
df.columns=['docid','url','title','body']
df.head()



def create_corpus(result):
  unique_docid=result['docid'].unique()
  condition=df['docid'].isin(unique_docid)
  corpus=df[condition].reset_index(drop=True)
  corpus=corpus.drop(columns='url')
  print('Number of Rows=>',len(corpus))
  return corpus

training_corpus=create_corpus(training_result)
training_corpus.head()

## Combining corpus and queries for training
training_corpus=training_corpus.rename(columns={'body':'text'})['text']
training_queries=training_queries.rename(columns={'query':'text'})['text']
combined_training=pd.concat([training_corpus,training_queries]).sample(frac=1).reset_index(drop=True)

#combined_training=pd.concat([training_corpus.rename(columns={'body':'text'})['text'],\
#                             training_queries.rename(columns={'query':'text'})['text']])\
#                             .sample(frac=1).reset_index(drop=True)


## Text processing

# Function for Cleaning Text
def clean_text(text):
    text=re.sub('\w*\d\w*','', text)
    text=re.sub('\n',' ',text)
    text=re.sub(r"http\s+", "", text)
    text=re.sub('[^a-z]',' ',text)
    return text

## Lowercasing the text
#training_corpus['cleaned']=training_corpus['body'].apply(lambda x:x.lower())
#testing_corpus['cleaned']=testing_corpus['body'].apply(lambda x:x.lower())
 
## Cleaning corpus using RegEx
#training_corpus['cleaned']=training_corpus['cleaned'].apply(lambda x: clean_text(x))
#testing_corpus['cleaned']=testing_corpus['cleaned'].apply(lambda x: clean_text(x))

## Removing extra spaces
#training_corpus['cleaned']=training_corpus['cleaned'].apply(lambda x: re.sub(' +',' ',x))
#testing_corpus['cleaned']=testing_corpus['cleaned'].apply(lambda x: re.sub(' +',' ',x))

## Removing Stopwords and Lemmatizing words
#training_corpus['lemmatized']=training_corpus['cleaned'].apply(lambda x: ' '.join([token.lemma_ for token in list(nlp(x)) if (token.is_stop==False)]))
#testing_corpus['lemmatized']=testing_corpus['cleaned'].apply(lambda x: ' '.join([token.lemma_ for token in list(nlp(x)) if (token.is_stop==False)]))


## Lowercasing the text
#training_queries['cleaned']=training_queries['query'].apply(lambda x:x.lower())
#testing_queries['cleaned']=testing_queries['query'].apply(lambda x:x.lower())

## Cleaning queries using RegEx
#training_queries['cleaned']=training_queries['cleaned'].apply(lambda x: clean_text(x))
#testing_queries['cleaned']=testing_queries['cleaned'].apply(lambda x: clean_text(x))

## Removing extra spaces
#training_queries['cleaned']=training_queries['cleaned'].apply(lambda x: re.sub(' +',' ',x))
#testing_queries['cleaned']=testing_queries['cleaned'].apply(lambda x: re.sub(' +',' ',x))




# Creating data for the model training
train_data=[]
for i in combined_training:
    train_data.append(i.split())

# Training a word2vec model from the given data set
w2v_model = Word2Vec(train_data, vector_size=300, min_count=2,window=5, sg=1,workers=4)

# Function returning vector reperesentation of a document
def get_embedding_w2v(doc_tokens):
    embeddings = []
    if len(doc_tokens)<1:
        return np.zeros(300)
    else:
        for tok in doc_tokens:
            if tok in w2v_model.wv.vocab:
                embeddings.append(w2v_model.wv.word_vec(tok))
            else:
                embeddings.append(np.random.rand(300))
        # mean the vectors of individual words to get the vector of the document
        return np.mean(embeddings, axis=0)

# Getting Word2Vec Vectors for Testing Corpus and Queries
testing_corpus['vector']=testing_corpus['lemmatized'].apply(lambda x :get_embedding_w2v(x.split()))
testing_queries['vector']=testing_queries['cleaned'].apply(lambda x :get_embedding_w2v(x.split()))

# Function for calculating average precision for a query
def average_precision(qid,qvector):
  
  # Getting the ground truth and document vectors
  qresult=testing_result.loc[testing_result['qid']==qid,['docid','rel']]
  qcorpus=testing_corpus.loc[testing_corpus['docid'].isin(qresult['docid']),['docid','vector']]
  qresult=pd.merge(qresult,qcorpus,on='docid')
  
  # Ranking documents for the query
  qresult['similarity']=qresult['vector'].apply(lambda x: cosine_similarity(np.array(qvector).reshape(1, -1),np.array(x).reshape(1, -1)).item())
  qresult.sort_values(by='similarity',ascending=False,inplace=True)

  # Taking Top 10 documents for the evaluation
  ranking=qresult.head(10)['rel'].values
  
  # Calculating precision
  precision=[]
  for i in range(1,11):
    if ranking[i-1]:
      precision.append(np.sum(ranking[:i])/i)
  
  # If no relevant document in list then return 0
  if precision==[]:
    return 0

  return np.mean(precision)

# Calculating average precision for all queries in the test set
testing_queries['AP']=testing_queries.apply(lambda x: average_precision(x['qid'],x['vector']),axis=1)

# Finding Mean Average Precision
print('Mean Average Precision=>',testing_queries['AP'].mean())


def ranking_ir(query):
  
  # pre-process Query
  query=query.lower()
  query=expand_contractions(query)
  query=clean_text(query)
  query=re.sub(' +',' ',query)

  # generating vector
  vector=get_embedding_w2v(query.split())

  # ranking documents
  documents=testing_corpus[['docid','title','body']].copy()
  documents['similarity']=testing_corpus['vector'].apply(lambda x: cosine_similarity(np.array(vector).reshape(1, -1),np.array(x).reshape(1, -1)).item())
  documents.sort_values(by='similarity',ascending=False,inplace=True)
  
  return documents.head(10).reset_index(drop=True)

ranking_ir('michael jordan')

print("end")