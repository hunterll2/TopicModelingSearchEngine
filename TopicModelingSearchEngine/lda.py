from gensim.corpora import Dictionary
from gensim.models import LdaModel
import time
import pickle

#import logging
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import preprocess
import constants

def train():
    # load data
    articles = pickle.load(open("dataset/"+constants.CLEANED_CORPUS_LIST, "rb"))

    # Create Dictionary
    dictionary = Dictionary(articles)

    # Create Term-Document Frequency matrix
    corpus = [dictionary.doc2bow(article) for article in articles]

    # Get user config data
    print("\nEnter config data:")
    num_topics = int(input("Number of topics>"))
    passes = int(input("Number of passes>"))
    alpha = float(input("Alpha value (1)>"))
    eta = float(input("Eta value (0.001)>"))

    # Saving options
    answer = input("\nSave the trained model as default? (y/n)>")

    if answer.lower() != "y":
        model_name = input("Enter model name >")

    # Start training the lda model
    print("\nStart training the LDA model")
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=passes, alpha=alpha, eta=eta)

    # Saving
    if answer.lower() == "y":
        lda.save("dataset/"+constants.LDA_MODEL)
    else:
        lda.save("_dataset/"+model_name)

    return

#articles = pickle.load(open("dataset/"+constants.CLEANED_CORPUS_LIST, "rb"))
#dictionary = Dictionary(articles)
#corpus = [dictionary.doc2bow(article) for article in articles]

#lda = LdaModel.load("dataset/"+constants.LDA_MODEL)

#for i in range(lda.num_topics):
#    print(lda.show_topic(i))



print()

