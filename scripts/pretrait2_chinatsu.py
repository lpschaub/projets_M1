#Chinatsu
import csv
from glob import glob
import os
from os import path
# from scripts.Evaluation import Evaluation
from PredictBOW_chinatsu import *
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
	if path.exists(out):
		with open(out,'a') as file:
			text = text.replace("\n","")
			writer = csv.writer(file,delimiter='\t')
			writer.writerow([text,label])
	else :
		with open(out,'a') as file:
			text = text.replace("\n","")
			writer = csv.writer(file,delimiter='\t')
			writer.writerow(["text","label"])
			writer.writerow([text,label])
  
   
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

	for fic in glob('../corpus/imdb/**/*.txt') : 
		file, label = getcontentlabel(fic)
		files2csv(file,"imdb.csv",label)
		
    
	print("ca marche ! bravo")
	