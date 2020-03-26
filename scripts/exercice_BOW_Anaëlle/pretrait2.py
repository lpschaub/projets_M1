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

def pipeline(file):
	corpus = pd.read_csv(file, sep='\t')
	corpus = corpus.reindex(np.random.permutation(corpus.index))
#	print(corpus['text'])
#	print(corpus['label'])
	corpus['text']=[elem.lower() for elem in corpus['text']]
	Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(corpus['text'],corpus['label'],test_size=0.2)
	Encoder = LabelEncoder()
	Train_Y = Encoder.fit_transform(Train_Y)
	Test_Y = Encoder.fit_transform(Test_Y)
	print(Train_Y)
	print(Test_Y)
	
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
    text=clean(text)
    with open(out, 'a') as f:
        f.write(f"{text}\t{label}\n")
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

def clean(ficstring) : 
    ficstring = ficstring.replace('(','')
    ficstring = ficstring.replace(')','')
    ficstring = ficstring.replace('\n',' ')
    ficstring = ficstring.replace('\t', ' ')
    ficstring = ficstring.replace('\r', ' ')
    ficstring = re.sub(r"<[^>]+>", " ", ficstring)
    return ficstring



if __name__ == '__main__':
	out = "./fichiercsv.tsv"
	with open(out, 'a') as f:
		f.write(f"text\tlabel\n")
	f.close()
	for fic in glob("../../corpus/imdb/**/*.txt"):
		label = getcontentlabel(fic)
		files2csv(label[0], out, label[1])
	pipeline(out)
