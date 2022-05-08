from gensim.corpora import Dictionary
from gensim.models import LdaModel

def train(df):
    # make list of docs
    articles = [x for x in df['cleaned']]

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