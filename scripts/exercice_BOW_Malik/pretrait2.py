from glob import glob
import os
import csv
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


def fList(corpusPath):
   filelist = [os.path.join(root, name)\
            for root, dirs, files in os.walk(corpusPath)\
            for name in files \
            if name.endswith(("txt"))]
   return(filelist)

def files2csv(corpusPath, out):
   fileList = fList(corpusPath)
   with open(out, "w") as f:
      writer = csv.writer(f, delimiter = "\t")
      writer.writerow(["TEXT", "LABEL"])
      for fileName in fileList:
         with open(fileName, "r") as review:
            text = review.read()
            if "pos" in fileName:
               label = "pos"
            else:
               label = "neg"
            writer.writerow([text, label.upper()])
    
   """
   à vous de compléter -> 
   passer d'une liste de fichiers contenant chacun un commentaire imdb à un seul fichier csv de la forme : 
   text   label ( c'est ce quon appelle l'en-tête)
   very good movie yeah POS
   I hated it ! NEG

   entre le text et le label c'est une tabulation 
   un commentaire par ligne (au besoin, enlever les retours chariot d'un commentaire)

   """





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
    corpusPath = "../corpus3/"
    files2csv(corpusPath, "./testCsv.csv")
