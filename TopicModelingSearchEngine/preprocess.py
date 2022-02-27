import nltk
import re
import spacy
import pickle
import logging
import pandas as pd

nlp = spacy.load('en_core_web_sm', disable=['ner','parser'])
nlp.max_length=5000000

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Tokenizer
from gensim.utils import simple_tokenize


###### Helpers Funcs

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

contractions_re=re.compile('(%s)' % '|'.join(contractions_dict.keys()))

def expand_contractions(text,contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, text)

def clean_text(text):
    text=re.sub('\w*\d\w*','', text)
    text=re.sub('\n',' ',text)
    text=re.sub(r"http\s+", "", text)
    text=re.sub('[^a-z]',' ',text)
    return text


###### Model Funcs

last_num = 50000;

def get_txts_cleand(ver = "new", num_docs = 1):
    if (ver == 'last'):
        return pickle.load(open("txts_cleand_"+str(last_num), "rb"))
    else:
        create_new(num_docs)
        return pickle.load(open("txts_cleand_"+str(last_num), "rb"))

def get_corpus(ver = 'new', num_docs = 1):
    if (ver == 'last'):
        return pickle.load(open("sample_corpus_"+str(last_num), "rb"))
    elif (len(ver)):
        return pickle.load(open(ver, "rb"))
    else:
        create_new(num_docs)
        return pickle.load(open("sample_corpus_"+str(last_num), "rb"))

def create_new(num):

    df = pd.read_table('./dataset/sample_corpus.tsv', header=None, encoding="latin1")
    df = df.iloc[:num]
    df.columns=['docid','url','title','body']
    df = df.dropna()

    df['cleaned']=df['body'].apply(lambda x:x.lower())
    df['cleaned']=df['cleaned'].apply(lambda x:expand_contractions(x))
    df['cleaned']=df['cleaned'].apply(lambda x: clean_text(x))
    df['cleaned']=df['cleaned'].apply(lambda x: re.sub(' +',' ',x))
    df['cleaned']=df['cleaned'].apply(lambda x: [token.lemma_ for token in list(nlp(x)) if token.is_stop==False])
    df['cleaned']=df['cleaned'].apply(lambda x: [token for token in x if len(token)>=3])
    df['cleaned']=df['cleaned'].apply(lambda x: ' '.join([str(e) for e in x]))

    txts = [list(simple_tokenize(x)) for x in df['cleaned'].values]

    pickle.dump( txts, open( "txts_cleand_"+str(num), "wb" ) )
    pickle.dump( df, open( "sample_corpus_"+str(num), "wb" ) )

    global last_num;
    last_num = num;

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