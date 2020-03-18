#a executer depuis le repertoire ou se trouve le script (exercice_BOW_Camille)

from glob import glob
import os
# from scripts.Evaluation import Evaluation
from PredictBOW_cam import *
#from Evaluation import Evaluation
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

   texte_clean=re.sub(r'(<[^>]*>)|\(|\)|\n|\t',' ',text)
   with open(out, 'a') as file:
       file.write(texte_clean+"\t"+label+"\n")





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

    with open("./csv_test.csv", 'w') as file:
        file.write("text\tlabel\n")
    for fic in glob('../../corpus/imdb/*/*.txt') :
        text,label=getcontentlabel(fic)
        files2csv(text,"./csv_test.csv",label)


    print("ca marche ! bravo")
