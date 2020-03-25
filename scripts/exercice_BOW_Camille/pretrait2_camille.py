#a executer depuis le repertoire ou se trouve le script (exercice_BOW_Camille)

from glob import glob
import os
import random
# from scripts.Evaluation import Evaluation
from PredictBOW_cam_V2 import *
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


def pipeline(fic):
    corpus=pd.read_csv(fic,sep='\t')
    corpus['text']=[elem.lower() for elem in corpus['text']]
    return corpus
    #print(corpus.head())
    #Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(corpus['text'],corpus['label'],test_size=0.2)
    #Encoder = LabelEncoder()
    #Train_Y = Encoder.fit_transform(Train_Y)


    #Test_Y = Encoder.fit_transform(Test_Y)
    #print(Train_Y)
    #print(Test_Y)

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

    #with open("./csv_test.csv", 'w') as file:
        #file.write("text\tlabel\n")

    #for fic in random.sample(glob('../../../imdb_test/*/*.txt'),len(glob('../../../imdb_test/*/*.txt'))) :
        #text,label=getcontentlabel(fic)
        #files2csv(text,"./csv_test.csv",label)

    corpus=CorpusCSV(pipeline("./csv_test.csv"))
    print(corpus.voc)
    corpus.getBOW()
    print(corpus.bow)
    print(corpus.bigram)
    print(corpus.trigram)
    corpus.getTFIDF()
    print(corpus.tfidf)

    print("ca marche ! bravo")
