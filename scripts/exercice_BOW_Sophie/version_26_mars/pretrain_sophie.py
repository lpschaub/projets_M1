from glob import glob
import os
from Evaluation import Evaluation
from predict_sophie import *
import spacy
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score, classification_report
import random
import sys

predicted = []
expected = []

nlp=spacy.load('en_core_web_sm')

def pipeline(file):
    corpus=pd.read_csv(file, sep='\t')
    
    return corpus

    print(f"\nPremières lignes du corpus :\n{corpus.head()}")
    corpus=corpus.reindex(np.random.permutation(corpus.index))
    print(f"\nPremières lignes du corpus après mélange :\n{corpus.head()}")
    print(f"\nTexte :\n{corpus['text']}")
    print(f"\nLabel :\n{corpus['label']}")
    
    corpus['text'].dropna(inplace=True)
    corpus['text'] = [entry.lower() for entry in corpus['text']]
    
    ### ...

    Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(corpus['text'], corpus['label'], test_size=0.2)
    
    print(f"\nTextes de train :\n{Train_X}\nLongueur : {len(Train_X)}")
    print(f"\nTextes de test :\n{Test_X}\nLongueur : {len(Test_X)}")
    print(f"\nLabels de train :\n{Train_Y}\nLongueur : {len(Train_Y)}")
    print(f"\nLabels de test :\n{Test_Y}\nLongueur : {len(Test_Y)}")
    
    Encoder = LabelEncoder()
    Train_Y = Encoder.fit_transform(Train_Y)
    Test_Y = Encoder.fit_transform(Test_Y)
    
    print(f"Labels de train : {Train_Y}")
    print(f"Labels de test : {Test_Y}")
    
    #sys.exit()
    
    representation = CorpusCSV(corpus['text'])
    voc = representation.voc
    
    #Train_X = CorpusCSV(Train_X, voc)
    #Test_X = CorpusCSV(Test_X, voc)
    #Train_X.getBOW()
    #Test_X.getBOW()
    #Train_X_bow = Train_X.bow
    #Test_X_bow = Test_X.bow    
    
def files2csv(nomfic):
    liste=[]
    for comm in glob('../corpus/imdb/*/*'):
        text=open(comm).read()
        text = text.replace('\n','')
        text = text.replace('\t','')
        text=text.lower()  
        if 'pos/' in comm:
            label="pos"
        else:
            label="neg"
        liste.append(f"{text}\t{label}\n")
        
    f = open(nomfic, 'w')
    f.write("text\tlabel\n")
    for comm in liste:
        f.write(comm)
    f.close
    
#def getFile(fic,pol): 
    #x = 0
    #for f in glob(fic+'/*') : 
        #x += 1
        #if x >= 2000 : 
            #os.system(f'mv {f} /home/schaub/Téléchargements/aclImdb_v1/aclImdb/test/{pol}/')

        #if x == 8000 : 
            #break

def getcontentlabel(file) :
    expect = '' 
    if 'pos/' in file : 
        expect = 'pos'
    else : 
        expect = 'neg'
    return open(file).read(),expect


def prediction(file,expect) : 
    pred = Predict(file, pw, nw, nlp)
    pred.predict()
    predicted.append(pred.predicted)
    expected.append(expect)


if __name__ == '__main__':
    print("Lancement de pretrain")
    #files2csv("commentaires.csv")
    #pipeline("commentaires.csv")
    corpus = pipeline("commentaires.csv")
    corpus = CorpusCSV(corpus)
    corpus.getBOW()
