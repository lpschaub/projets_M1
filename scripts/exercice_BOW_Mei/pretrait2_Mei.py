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

import re

def pipeline(file) : 

    Corpus = pd.read_csv(file, sep='\t')
    corpus = Corpus.reindex(np.random.permutation(Corpus.index))
    # print(corpus.head())
    # # Step - a : Remove blank rows if any.
    # print(corpus['text'])
    # print(corpus['label'])

    corpus['text'].dropna(inplace=True)# Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
    
    # print(type(Corpus['text']))
    # for elem in Corpus['text'] : 
    #     print(elem)
    #     print(type(elem))
    #     break
    corpus['text'] = [entry.lower() for entry in corpus['text']]
    Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(corpus['text'],corpus['label'],test_size=0.2)
    # print(len(Train_X))
    # print(len(Train_Y))

    Encoder = LabelEncoder()
    Train_Y = Encoder.fit_transform(Train_Y)
    print(Train_Y)

    # Test_Y = Encoder.fit_transform(Test_Y)
    # print(Test_Y)
    # sys.exit()


    representation = CorpusCSV(corpus['text'])
    voc = representation.voc
    
# def cleanlines(line):  #remplacer tous les ponctuations et tous les chiffres par une espace
#         p1=re.compile(r'[(][: @ . , ？！\s][)]')
#         p2=re.compile(r'[「『]')
#         p3=re.compile(r'[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）0-9 , : ; \ \[\ \]\ ]')
#         line=p1.sub(r' ',line)
#         line=p2.sub(r' ',line)
#         line=p3.sub(r' ',line)
#         return line

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
           chaine = ficstring.replace('\t',' ')
           chaine = re.sub(r'<[^>]+>', '', chaine) 
           chaine = re.sub(r'[\(\[,\.\?!\*\]\)"\'/0-9]',' ',chaine)
           chaine = re.sub(r' +',' ',chaine)
           chaine = chaine.lower()
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
    
    f = open('out.csv','w')
    f.write(f"text\tlabel\n")
    f.close()
    files2csv('../corpus/imdb/neg/*','out.csv','NEG')
    files2csv('../corpus/imdb/pos/*','out.csv','POS')
    
    # pipeline('out.csv')
    print("ca marche ! bravo")