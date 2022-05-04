import re
import spacy
nlp = spacy.load("en_core_web_sm")
import pickle
import constants as c

# Module functions
contractions_re = re.compile('(%s)' % '|'.join(c.CONTRACTIONS_DICT.keys()))
def expand_contractions(text, contractions_dict = c.CONTRACTIONS_DICT):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, text)

def clean_text(text):
    text = re.sub('\w*\d\w*','', text)
    text = re.sub('\n',' ',text)
    text = re.sub(r"http\s+", "", text)
    text = re.sub('[^a-z]',' ',text)
    return text

def clean(txt):
    cleand = []
    txt = txt.lower()
    txt = clean_text(txt)
    
    for token in list(nlp(txt)):
        clean_token = token.lemma_
            
        if (token.is_stop): continue
        if (token.is_space): continue
        if (len(token) < 3): continue

        cleand.append(clean_token)

    return ' '.join([str(e) for e in cleand])

def create_new(df, num = 50000):
    df = df.iloc[:num]
    df.columns=['docid','url','title','body']
    df = df.dropna()

    df['cleaned']=df['body'].apply(lambda x: x.lower())
    df['cleaned']=df['cleaned'].apply(lambda x: expand_contractions(x))
    df['cleaned']=df['cleaned'].apply(lambda x: clean_text(x))
    df['cleaned']=df['cleaned'].apply(lambda x: re.sub(' +',' ',x))
    df['cleaned']=df['cleaned'].apply(lambda x: [token.lemma_ for token in list(nlp(x)) if token.is_stop==False])
    df['cleaned']=df['cleaned'].apply(lambda x: [token for token in x if len(token)>=3])
    df['cleaned']=df['cleaned'].apply(lambda x: ' '.join([str(e) for e in x]))

    pickle.dump( df, open( "dataset/"+c.CLEANED_CORPUS_TABLE, "wb" ) )

    return