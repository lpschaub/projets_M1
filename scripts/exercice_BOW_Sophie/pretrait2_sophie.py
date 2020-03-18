from glob import glob
import os
# from scripts.Evaluation import Evaluation
from PredictBOW import *
from Evaluation import Evaluation
import spacy
predicted = []
expected = []


nlp=spacy.load('en_core_web_sm')
# from Predict import Predict


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



def files2csv(nomfichier):
    """
    à vous de compléter -> 
    passer d'une liste de fichiers contenant chacun un commentaire imdb à un seul fichier csv de la forme : 
    text   label ( c'est ce quon appelle l'en-tête)
    very good movie yeah POS
    I hated it ! NEG
    entre le text et le label c'est une tabulation 
    un commentaire par ligne (au besoin, enlever les retours chariot d'un commentaire)
    """
    
    f = open(nomfichier, 'w')
    f.write("text\tlabel\n")
    for comm in glob('../corpus/imdb/pos/*'):
        text = open(comm).read()
        text = text.replace('\n','')
        text = text.replace('\t','')
        f.write(f"{text}\tpos\n") 
    for comm in glob('../corpus/imdb/neg/*'):
        text = open(comm).read()
        text = text.replace('\n','')
        text = text.replace('\t','')
        f.write(f"{text}\tneg\n") 
    f.close() 


def getFile(fic,pol): 
    x = 0
    for f in glob(fic+'/*') : 
        x += 1
        if x >= 2000 : 
            os.system(f'mv {f} /home/schaub/Téléchargements/aclImdb_v1/aclImdb/test/{pol}/')

        if x == 8000 : 
            break


def getcontentlabel(file) :
    
    expect = '' 
    if 'p/' in file : 
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

    print("ca marche ! bravo")
    files2csv("commentaires.csv")