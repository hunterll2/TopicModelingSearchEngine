import re
import pickle
import pandas as pd
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
        clean_token = token.lemma_
            
        if (token.is_stop): continue
        if (token.is_space): continue
        if (len(token) < 3): continue

        cleand.append(clean_token)

    # convert the list of words into one string
    return ' '.join([str(e) for e in cleand])

def create_new(df, num = 50000):
    # take a portion of the dataset
    df = df.iloc[:num]
    df.columns=['docid','url','title','body']
    df = df.dropna()

    # for every doucment text perform the clean preprocess
    df['cleaned'] = df['body'].apply(lambda x: clean(x))

    # save the new cleaned dataset
    pickle.dump( df, open( "dataset/cleaned_corpus_table", "wb" ) )

    return