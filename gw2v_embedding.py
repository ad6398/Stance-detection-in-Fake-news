# run and tested
#this script returns Google's word2vec embedding of texts
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import string
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import numpy as np

filename = 'GoogleNews-vectors-negative300.bin'
G_w2v = KeyedVectors.load_word2vec_format(filename, binary=True)

def pre_process(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    return text

def make_embed(text):
    feature= np.zeros((len(text),300),dtype= object)
    c=0
    for wrd in text:
        if wrd in G_w2v:
            feature[c]=G_w2v[wrd].reshape(1,300)
            c=c+1
    res= np.zeros(300)
    for i in range(300):
        res[i]=0
        for j in range(len(feature)):
            res[i]= res[i] + feature[j,i]
        res[i]= res[i]/(len(feature))
    return res

def gw2v_sentence_embeddings(text):
    clean_text= pre_process(text)
    return make_embed(clean_text)
