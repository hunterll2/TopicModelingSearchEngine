
import pandas as pd
import numpy as np
import dask.dataframe as dd
from tabulate import tabulate
import re
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import spacy

# Queries

queries_train=pd.read_table('./dataset/queries.doctrain.tsv',header=None)
queries_train.columns=['qid','query']
print('Shape=>',queries_train.shape)
queries_train.head()

queries=queries_train.sample(n=2000,random_state=42).reset_index(drop=True)
print('Shape=>',queries.shape)
queries.head()

#df = pd.DataFrame(queries)
#print(tabulate(df, headers = 'keys', tablefmt = 'psql'))

# Creating Training Set of Queries
training_queries=queries.iloc[:1000]
print('Shape=>',training_queries.shape)
training_queries.head()

# Creating Testing Set of Queries
testing_queries=queries.iloc[1000:]
print('Shape=>',testing_queries.shape)
testing_queries.head()

# Documents

train_top100=pd.read_table('./dataset/msmarco-doctrain-top100',delimiter=' ',header=None)
train_top100.columns=['qid','Q0','docid','rank','score','runstring']
print('Shape=>',train_top100.shape)
train_top100.head()

# Reducing train_top100 for training
training_ranked100=train_top100[train_top100['qid'].isin(training_queries['qid'].unique())].reset_index(drop=True)
print('Shape=>',training_ranked100.shape)
training_ranked100.head()

# Reducing train_top100 for testing
testing_ranked100=train_top100[train_top100['qid'].isin(testing_queries['qid'].unique())].reset_index(drop=True)
print('Shape=>',testing_ranked100.shape)
testing_ranked100.head()

# Labelling Top 10 as 1 and last 10 as 0
rel=list(range(1,11))
nonrel=list(range(91,101))
training_ranked100['rel']=training_ranked100['rank'].apply(lambda x: 1 if x in rel else ( 0 if x in nonrel else np.nan))
testing_ranked100['rel']=testing_ranked100['rank'].apply(lambda x: 1 if x in rel else ( 0 if x in nonrel else np.nan))

# Result set for Training
training_result=training_ranked100.dropna()
training_result['rel']=training_result['rel'].astype(int)
print('Shape=>',training_result.shape)
training_result.head()

# Result set for Testing
testing_result=testing_ranked100.dropna()
testing_result['rel']=testing_result['rel'].astype(int)
print('Shape=>',testing_result.shape)
testing_result.head()

# corpus 

df=pd.read_table('./dataset/sample_corpus.tsv',header=None,encoding="latin1")
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

testing_corpus=create_corpus(testing_result)
testing_corpus.head()

## Text processing

# Lowercasing the text
training_corpus['cleaned']=training_corpus['body'].apply(lambda x:x.lower())
testing_corpus['cleaned']=testing_corpus['body'].apply(lambda x:x.lower())

## Dictionary of english Contractions
contractions_dict = { "ain't": "are not","'s":" is","aren't": "are not","can't": "can not","can't've": "cannot have",
"'cause": "because","could've": "could have","couldn't": "could not","couldn't've": "could not have",
"didn't": "did not","doesn't": "does not","don't": "do not","hadn't": "had not","hadn't've": "had not have",
"hasn't": "has not","haven't": "have not","he'd": "he would","he'd've": "he would have","he'll": "he will",
"he'll've": "he will have","how'd": "how did","how'd'y": "how do you","how'll": "how will","i'd": "i would",
"i'd've": "i would have","i'll": "i will","i'll've": "i will have","i'm": "i am","i've": "i have",
"isn't": "is not","it'd": "it would","it'd've": "it would have","it'll": "it will","it'll've": "it will have",
"let's": "let us","ma'am": "madam","mayn't": "may not","might've": "might have","mightn't": "might not",
"mightn't've": "might not have","must've": "must have","mustn't": "must not","mustn't've": "must not have",
"needn't": "need not","needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not",
"oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not",
"shan't've": "shall not have","she'd": "she would","she'd've": "she would have","she'll": "she will",
"she'll've": "she will have","should've": "should have","shouldn't": "should not",
"shouldn't've": "should not have","so've": "so have","that'd": "that would","that'd've": "that would have",
"there'd": "there would","there'd've": "there would have",
"they'd": "they would","they'd've": "they would have","they'll": "they will","they'll've": "they will have",
"they're": "they are","they've": "they have","to've": "to have","wasn't": "was not","we'd": "we would",
"we'd've": "we would have","we'll": "we will","we'll've": "we will have","we're": "we are","we've": "we have",
"weren't": "were not","what'll": "what will","what'll've": "what will have","what're": "what are",
"what've": "what have","when've": "when have","where'd": "where did",
"where've": "where have","who'll": "who will","who'll've": "who will have","who've": "who have",
"why've": "why have","will've": "will have","won't": "will not","won't've": "will not have",
"would've": "would have","wouldn't": "would not","wouldn't've": "would not have","y'all": "you all",
"y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
"you'd": "you would","you'd've": "you would have","you'll": "you will","you'll've": "you will have",
"you're": "you are","you've": "you have"}

## Regular expression for finding contractions
contractions_re=re.compile('(%s)' % '|'.join(contractions_dict.keys()))

## Function for expanding contractions
def expand_contractions(text,contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, text)

# Expanding Contractions
training_corpus['cleaned']=training_corpus['cleaned'].apply(lambda x:expand_contractions(x))
testing_corpus['cleaned']=testing_corpus['cleaned'].apply(lambda x:expand_contractions(x))

## Function for Cleaning Text
def clean_text(text):
    text=re.sub('\w*\d\w*','', text)
    text=re.sub('\n',' ',text)
    text=re.sub(r"http\s+", "", text)
    text=re.sub('[^a-z]',' ',text)
    return text
 
# Cleaning corpus using RegEx
training_corpus['cleaned']=training_corpus['cleaned'].apply(lambda x: clean_text(x))
testing_corpus['cleaned']=testing_corpus['cleaned'].apply(lambda x: clean_text(x))

# Removing extra spaces
training_corpus['cleaned']=training_corpus['cleaned'].apply(lambda x: re.sub(' +',' ',x))
testing_corpus['cleaned']=testing_corpus['cleaned'].apply(lambda x: re.sub(' +',' ',x))

# Stopwords removal & Lemmatizing tokens using SpaCy

nlp = spacy.load('en_core_web_sm',disable=['ner','parser'])
nlp.max_length=5000000

# Removing Stopwords and Lemmatizing words
training_corpus['lemmatized']=training_corpus['cleaned'].apply(lambda x: ' '.join([token.lemma_ for token in list(nlp(x)) if (token.is_stop==False)]))
testing_corpus['lemmatized']=testing_corpus['cleaned'].apply(lambda x: ' '.join([token.lemma_ for token in list(nlp(x)) if (token.is_stop==False)]))

# Lowercasing the text
training_queries['cleaned']=training_queries['query'].apply(lambda x:x.lower())
testing_queries['cleaned']=testing_queries['query'].apply(lambda x:x.lower())

# Expanding contractions
training_queries['cleaned']=training_queries['cleaned'].apply(lambda x:expand_contractions(x))
testing_queries['cleaned']=testing_queries['cleaned'].apply(lambda x:expand_contractions(x))

# Cleaning queries using RegEx
training_queries['cleaned']=training_queries['cleaned'].apply(lambda x: clean_text(x))
testing_queries['cleaned']=testing_queries['cleaned'].apply(lambda x: clean_text(x))

# Removing extra spaces
training_queries['cleaned']=training_queries['cleaned'].apply(lambda x: re.sub(' +',' ',x))
testing_queries['cleaned']=testing_queries['cleaned'].apply(lambda x: re.sub(' +',' ',x))


# Combining corpus and queries for training

combined_training=pd.concat([training_corpus.rename(columns={'lemmatized':'text'})['text'],\
                             training_queries.rename(columns={'cleaned':'text'})['text']])\
                             .sample(frac=1).reset_index(drop=True)

# Creating data for the model training
train_data=[]
for i in combined_training:
    train_data.append(i.split())

# Training a word2vec model from the given data set
w2v_model = Word2Vec(train_data, vector_size=300, min_count=2,window=5, sg=1,workers=4)
#w2v_model.save("w2v_model")
#w2v_model = Word2Vec.load("w2v_model")

# Function returning vector reperesentation of a document
def get_embedding_w2v(doc_tokens):
    embeddings = []
    if len(doc_tokens)<1:
        return np.zeros(300)
    else:
        for tok in doc_tokens:
            if tok in w2v_model.wv.index_to_key:
                embeddings.append(w2v_model.wv.word_vec(tok))
            else:
                embeddings.append(np.random.rand(300))
        # mean the vectors of individual words to get the vector of the document
        return np.mean(embeddings, axis=0)

# Getting Word2Vec Vectors for Testing Corpus and Queries
testing_corpus['vector']=testing_corpus['lemmatized'].apply(lambda x :get_embedding_w2v(x.split()))
testing_queries['vector']=testing_queries['cleaned'].apply(lambda x :get_embedding_w2v(x.split()))

# Function for calculating average precision for a query
#def average_precision(qid,qvector):
  
#  # Getting the ground truth and document vectors
#  qresult=testing_result.loc[testing_result['qid']==qid,['docid','rel']]
#  qcorpus=testing_corpus.loc[testing_corpus['docid'].isin(qresult['docid']),['docid','vector']]
#  qresult=pd.merge(qresult,qcorpus,on='docid')
  
#  # Ranking documents for the query
#  qresult['similarity']=qresult['vector'].apply(lambda x: cosine_similarity(np.array(qvector).reshape(1, -1),np.array(x).reshape(1, -1)).item())
#  qresult.sort_values(by='similarity',ascending=False,inplace=True)

#  # Taking Top 10 documents for the evaluation
#  ranking=qresult.head(10)['rel'].values
  
#  # Calculating precision
#  precision=[]
#  for i in range(1,11):
#    if ranking[i-1]:
#      precision.append(np.sum(ranking[:i])/i)
  
#  # If no relevant document in list then return 0
#  if precision==[]:
#    return 0

#  return np.mean(precision)

## Calculating average precision for all queries in the test set
#testing_queries['AP']=testing_queries.apply(lambda x: average_precision(x['qid'],x['vector']),axis=1)

## Finding Mean Average Precision
#print('Mean Average Precision=>',testing_queries['AP'].mean())


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

print(ranking_ir('michael jordan'))

print("end")