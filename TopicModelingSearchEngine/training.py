import nltk
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, LdaModel
from multiprocessing import Pool
import sqlite3 as sql
import pandas as pd
import numpy as np
import logging
import time
import re

# Tokenizer
from gensim.utils import simple_tokenize

# NLTK Stop words
from nltk.corpus import stopwords
stopwords = nltk.corpus.stopwords.words("english")

# Stermmer
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

# Dataset
db = 'dataset/enwiki-20170820.db'

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Helpers
def get_query(select, db=db):
    '''
    1. Connects to SQLite database (db)
    2. Executes select statement
    3. Return results and column names
    
    Input: 'select * from analytics limit 2'
    Output: ([(1, 2, 3)], ['col_1', 'col_2', 'col_3'])
    '''
    with sql.connect(db) as conn:
        c = conn.cursor()
        c.execute(select)
        col_names = [str(name[0]).lower() for name in c.description]
    return c.fetchall(), col_names

def get_article(article_id):
    '''
    1. Construct select statement
    2. Retrieve all section_texts associated with article_id
    3. Join section_texts into a single string (article_text)
    4. Return
    
    Input: 100
    Output: 'the austroasiatic languages in ...'
    '''
    select = '''select section_text from articles where article_id=%d''' % article_id
    docs, _ = get_query(select)
    docs = [doc[0] for doc in docs]
    doc = ''.join(docs)
    return doc

# Get articles ids
select = '''select distinct article_id from articles'''
article_ids, _ = get_query(select)
article_ids = [article_id[0] for article_id in article_ids]
sample_article_ids = article_ids[:100]

# Fetch each article text and perform preprocess
articles = []
for id in sample_article_ids:
    article = get_article(id)
    article = stemmer.stem(article)
    article = list(simple_tokenize(article))
    article = [word for word in article if word not in stopwords]
    articles.append(article)

# Create Dictionary
dictionary = Dictionary(articles)
#dictionary.save('dictionary10')
#dictionary = Dictionary.load('dictionary10')

# Create Term-Document Frequency matrix
corpus = [dictionary.doc2bow(article) for article in articles]

# Start training the lda model
start = time.time()
lda = LdaModel(corpus=corpus, num_topics=50, id2word=dictionary, passes=1, alpha=1, eta=0.001)
#lda.save('lda10')
#lda = LdaModel.load('lda10');
end = time.time()
print('Time to train LDA from generator: %0.2fs' % (end - start))

## Print topics with their words
topics = lda.show_topics()

for topic in topics:
    print("Topic ", topic[0])
    topic_terms = lda.show_topic(topic[0])
    terms = []
    for term, weight in topic_terms:
        print(term, weight)
        terms.append(term)
    print(terms)
    print("\n")

##
topic_term = lda.get_topics()