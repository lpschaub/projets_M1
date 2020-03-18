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
        
class Vocab(object):
    def __init__(self,corpus) :
        self.corpus = corpus 
        self.voc = set()
         
    def add_words(self, doc, vocab):
        words = doc.split(" ")       
        for word in words:
            word_fmt = word.strip(",?;.:!\(\)<>")
            vocab.add(word_fmt.lower())
            
    def build_voc(self):
        for doc in corpus:
            docstr = open(doc).read()
            self.add_words(docstr,self.voc)
            
    def preprocessing(self, doc):
        list_word=set()
        for token in doc:
            word_fmt = token.text.strip(",?;.:!\(\)<>")
            list_word.add(word_fmt.lower())
        
        return list_word
        
    def conditional_test(self, doc_list):
        new_list =[]
        for word in self.voc:
            if word in doc_list:
                new_list.append(1)
            else:
                new_list.append(0)
                
        return new_list
        
        
class Predict(object) : 

    def __init__(self, doc, positive_words, negative_words, seuil = 0.5) : 
       self.doc = doc
       self.positive_words = positive_words
       self.negative_words = negative_words
       self.seuil = seuil
       self.voc = voc
       self.bow = [] #liste des listes 
           

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

    
def ReadCorpus(paths):
    corpus=set()
    for path in paths:
        for file in glob(os.path.join(path, "*.txt")):
            lines = open(file).read(). splitlines()
            for line in lines:
               add_word(doc, corpus)
     
    return corpus
        
if __name__ == '__main__':

    # Reading of the positive words corpus
    #positive_words = set(open("positive-words.txt").read(). splitlines()) 
    # Reading of the negative words corpus
    #negative_words = set(open("negative-words.txt").read(). splitlines()) 
    # path of the tested files
    paths = ["projets_M1/corpus/imdb/pos/","projets_M1/corpus/imdb/neg/"]
    corpus = [file for path in paths for file in glob(os.path.join(path, "*.txt"))]
    vocab = Vocab(corpus)
    vocab.build_voc()
    
    big_list=[]
    for path in paths:
        for file in glob(os.path.join(path, "*.txt")):
            # Reading of the tested file
            nlp = spacy.load('en_core_web_sm')
            doc = nlp(open(file).read())
            #Creation of the emotion prediction object
            #pred = Predict(doc, corpus)
            #pred.predict()
            #print(pred.predicted + " => "+ file  + "(" + str(pred.seuil) +")") 
            doc_list = vocab.preprocessing(doc)
            conditional_doc = vocab.conditional_test(doc_list)
            big_list.append(conditional_doc)
print(big_list)