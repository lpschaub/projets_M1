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



def files2csv(text,out,label):
    
   """
   à vous de compléter -> 
   passer d'une liste de fichiers contenant chacun un commentaire imdb à un seul fichier csv de la forme : 
   text   label ( c'est ce quon appelle l'en-tête)
   very good movie yeah POS
   I hated it ! NEG

   entre le text et le label c'est une tabulation 
   un commentaire par ligne (au besoin, enlever les retours chariot d'un commentaire)

   """
   with open(out,'a+',encoding = 'utf-8') as f:
       for fic in glob(text):
           ficstring = open(fic).read()
           chaine = ficstring.replace('\n',' ')
           f.write(f'{chaine}\t{label}\n')




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

    files2csv('../corpus/imdb/neg/*','out.csv','NEG')
    files2csv('../corpus/imdb/pos/*','out.csv','POS')
    print("ca marche ! bravo")