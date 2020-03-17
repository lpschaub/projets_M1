import re
import spacy 
from spacy import displacy
from glob import glob
import os
#from math import exp, expml
nlp = spacy.load('en_core_web_sm')

def LoadCorpusFile(file_path):
    with open ("positive-words.txt", 'r') as f:
        return set(f.read().splitlines())

class Predict(object) : 

    def __init__(self, doc, positive_words, negative_words, seuil = 0.5) : 
       self.doc = doc
       self.positive_words = positive_words
       self.negative_words = negative_words
       self.seuil = seuil

    def workspace(self):
       for token in self.doc:
            token_lower = token.text.lower()
            if token_lower in self.positive_words:
                self.seuil += 0.1
            elif token_lower in self.negative_words :
                self.seuil -= 0.1

    def predict(self) :
        self.workspace()
        if self.seuil < 0.5 : 
            self.predicted = 'neg'
        else : 
            self.predicted = 'pos'


if __name__ == '__main__':

    # Reading of the positive words corpus
    positive_words = set(open("positive-words.txt").read(). splitlines()) 
    # Reading of the negative words corpus
    negative_words = set(open("negative-words.txt").read(). splitlines()) 
    # path of the tested files
    path = "projets_M1/corpus/imdb/pos/"
    
    for file in glob(os.path.join(path, "*.txt")):
    # Reading of the tested file
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(open(file).read())
        #Creation of the emotion prediction object
        pred = Predict(doc, positive_words, negative_words)
        pred.predict()
        print(pred.predicted + " => "+ file  + "(" + str(pred.seuil) +")") 
 