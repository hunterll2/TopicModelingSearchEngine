import re
import spacy
nlp = spacy.load("en_core_web_sm")

import constants as c

# Module functions
contractions_re = re.compile('(%s)' % '|'.join(c.CONTRACTIONS_DICT.keys()))

def expand_contractions(text, contractions_dict = c.CONTRACTIONS_DICT):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, text)

def clean(txt):
    # Convert the text into lower case
    txt = txt.lower()

    # convert contractions into its original form
    txt = expand_contractions(txt)

    # remove digits
    txt = re.sub('\w*\d\w*', '', txt)

    # remove empty lins
    txt = re.sub('\n', ' ', txt)

    # remove brackets, links, random letters
    txt = re.sub('[^a-z]', ' ', txt)
    
    # normalize each word and exclude stops, spaces, and characters
    cleand = []
    for token in list(nlp(txt)):
        if (token.is_stop): continue
        if (token.is_space): continue
        if (len(token) < 3): continue

        cleand.append(token.lemma_)

    # convert the list of words into one string
    return ' '.join([e for e in cleand])

def create_new(corpus, num = 50000):
    print("\n# Start preprocess documents.")

    ## 1. Preparare corpus

    # take a portion of the dataset
    corpus = corpus.iloc[:num]

    # Set columns names ([0, 1, 2, 3] => ['id','url','title','body'])
    corpus.columns = ['docid','url','title','body']

    # Remove empty rows
    corpus = corpus.dropna()
    
    # Correct index numbers
    corpus.reset_index(drop=True)

    
    ## 2. Perform Text-Processing

    # for every doucment text: perform the clean preprocess
    corpus['cleaned'] = corpus['body'].apply(lambda x: clean(x))

    # for every document text: convert text into list of words ("simple text" => ["simple", "text"])
    corpus['list'] = corpus['cleaned'].apply(lambda x: x.split())

    # done
    print("* Documents processed successfully.")
    return corpus