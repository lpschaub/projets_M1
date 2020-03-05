# -*- coding: utf-8 -*-
#Anaëlle Pierredon

import spacy
from spacy import displacy

from stop_words import get_stop_words

def MaxFreq(valeur):
    m=0
    for item in valeur:
        nb = item[1]
        if nb > m:
            o = item
    return(item)

index= {}
liste = []
stop_words = get_stop_words('fr')
nlp = spacy.load('fr_core_news_sm')

documents = {
    "doc1" : "La bouffe est pas mauvaise",
    "doc2" : "Si, je le trouve pas bon",
    "doc3" : "J'avoue il est vraiment mauvais"
    }

for item in documents :
    doc = nlp(documents[item])                 #tokenisation
    for token in doc:
        token = token.lemma_                    #lemmatisation
        if token in stop_words:                #stopwordisation
            continue
        if token in [".", ",","!","?",":","-",'"',"'"]:
            continue
        else:
            if token in index:
                n = len(index[token])-1
                if item == index[token][n][0] :
                    index[token][n] = (index[token][n][0], index[token][n][1]+1)
                else:
                    index[token].append((item , 1))
                
            else:
                index[token] = [(item , 1)]
        
for cle, valeur in index.items():
    print(f"{cle} : {valeur}")
    print(f"Fréquence maximale : {MaxFreq(valeur)} \n")
    
            
    
