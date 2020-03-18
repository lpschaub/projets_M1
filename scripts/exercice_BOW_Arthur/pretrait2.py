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



def file2csv(text, label):
	
	with open("en-tete.csv", 'w', encoding='utf-8') as file :
		for item, item2 in zip(text, label) :
			file.write(f"{item}\t{item2}\n")



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
    if '/pos/' in file : 
        expect = 'pos'
    else : 
        expect = 'neg'

    return open(file).read(),expect


def prediction(file,expect) : 


    pred = Predict(file, pw, nw, nlp)
    pred.predict()
    predicted.append(pred.predicted)
    expected.append(expect)

def clean(ficstring) : 
	exception = [".", ";", ":", ",", "!", "?", "(", ")", ">", "<", "\"", "...", "/", "-", "_", "\t", ""]
	ficstring = re.sub("<br|/>", '', ficstring)
	for item in exception :
		ficstring = ficstring.replace(item,'')
	ficstring = ficstring.replace('\n',' ')
	return ficstring

if __name__ == '__main__':

		text = []
		label = []
		for file in glob('../corpus/imdb/**/*.txt') :
			content = getcontentlabel(file)
			text.append(clean(content[0]))
			label.append(content[1])
		file2csv(text, label)
    