import re
import numpy as np
import pandas as pd
from glob import glob
from collections import Counter
import spacy
from spacy import displacy
nlp = spacy.load("en_core_web_sm")

class Voc(object):
    def __init__(self, corpus):
        self.corpus = corpus
        self.vocMonograms = []
        self.vocBigrams = []
        self.vocTrigrams = []
    def build_voc(self):
        self.IDF = Counter()
        #self.fileToken = []
        self.fileCount = 0
        for fic in self.corpus.lire()['TEXT']:
            self.docToken = []
            self.fileCount += 1
            ficstring = self.clean(fic.replace('\n',' '))
            doc = nlp(ficstring)
            for sent in doc.sents:
                text = [token.orth_ for token in sent if not token.is_punct\
                        and not token.is_space]
                for i in range(len(text)):
                    token = str(text[i]).lower()
                    if (token not in self.docToken):
                        self.docToken.append(token)
                    if(token not in self.vocMonograms):
                        self.vocMonograms.append(token)
                    if(i+1 < len(text)):
                        bigram = (str(text[i]).lower(), str(text[i+1]).lower())
                        if (bigram not in self.docToken):
                            self.docToken.append(bigram)
                        if(bigram not in self.vocBigrams):
                            self.vocBigrams.append(bigram)
                        if (i+2 < len(text)):
                            trigram = (str(text[i]).lower(), \
                                       str(text[i+1]).lower(), \
                                       str(text[i+2]).lower())
                            if (trigram not in self.docToken):
                                self.docToken.append(trigram)
                            if(trigram not in self.vocTrigrams):
                                self.vocTrigrams.append(trigram)
            #self.fileToken = self.fileToken + self.docToken
            for token in set(self.docToken):
                self.IDF[token] += 1
        for key, value in self.IDF.items():
            df = value/self.fileCount
            if (key in self.vocMonograms and (df <= 0.2 or df >= 0.8)):
                self.vocMonograms.remove(key)
            elif (key in self.vocBigrams and (df <= 0.1 or df >= 0.9)):
                self.vocBigrams.remove(key)
            elif (key in self.vocTrigrams and (df < 0.005 or df > 0.99)):
                self.vocTrigrams.remove(key)
            else:
                self.IDF[key] = (self.fileCount/value)
                
        #print(self.IDF)
        print("Voc monogrammes:")
        print(len(self.vocMonograms))
        print("Voc bigrammes:")
        print(len(self.vocBigrams))
        print("Voc trigrammes:")
        print(len(self.vocTrigrams))
        print(self.vocMonograms)
        print(self.vocBigrams)
        print(self.vocTrigrams)
        return (self.vocMonograms, self.vocBigrams, self.vocTrigrams, self.IDF)
    def clean(self, ficstring):
        #patt = re.compile('<br />')
        patt = re.compile('<[^>]+>')
        ficstring = ficstring.replace('(', '')
        ficstring = ficstring.replace(')', '')
        ficstring = re.sub(patt, ' ', ficstring)
        ficstring = re.sub('\.', r'. ', ficstring)
        return ficstring
    def ecrire_voc(self, out):
        out.write('\n'.join([elem for elem in self.voc]))
    def load_voc(self, vocfile):
        self.voc = Voc.string(vocfile)




class Corpus(object):
    def __init__(self, path, voc = ""):
        self.corpus = path
        if voc:
            self.voc = Voc.load_voc(voc)
        else:
            voc = Voc(self)
            voc.build_voc()
            self.vocMonograms = voc.vocMonograms
            self.vocBigrams = voc.vocBigrams
            self.vocTrigrams = voc.vocTrigrams
            self.IDF = voc.IDF
        self.bowM = []
        self.bowB = []
        self.bowT = []
    def lire(self):
        return pd.read_csv(open(self.corpus), delimiter="\t")
    def getBOW(self):
        x = 0
        patt = re.compile('<[^>]+>')
        for fic in self.lire()['TEXT']:
            self.currVocM = []
            self.currVocB = []
            self.currVocT = []
            currText = fic
            currText = currText.replace('\n',' ')
            currText = currText.replace('(', '')
            currText = currText.replace(')', '')
            currText = re.sub(patt, ' ', currText)
            currText = re.sub('\.', r'. ', currText)
            doc = nlp(currText)
            for sent in doc.sents:
                text = [token.orth_ for token in sent if not token.is_punct\
                        and not token.is_space]
                for i in range(len(text)):
                    token = str(text[i]).lower()
                    self.currVocM.append(token)
                    if(i+1 < len(text)):
                        bigram = (str(text[i]).lower(), str(text[i+1]).lower())
                        self.currVocB.append(bigram)
                        if (i+2 < len(text)):
                            trigram = (str(text[i]).lower(), \
                                       str(text[i+1]).lower(), \
                                       str(text[i+2]).lower())
                            self.currVocT.append(trigram)
            #print(currVoc)
            vec = []
            #print(self.currVocM)
            for word in self.vocMonograms:
                if word in set(self.currVocM):
                    TF = (self.currVocM.count(word))/len(self.currVocM)
                    TFIDF = TF * np.log(self.IDF[word])
                    vec.append(TFIDF)
                else:
                    vec.append(0)
            #print(vec)
            self.bowM.append(vec)
            vec = []
            #print(self.currVocB)
            for bigram in self.vocBigrams:
                if bigram in set(self.currVocB):
                    TF = (self.currVocB.count(bigram))/len(self.currVocB)
                    TFIDF = TF * np.log(self.IDF[bigram])
                    vec.append(TFIDF)
                else:
                    vec.append(0)
            self.bowB.append(vec)
            vec = []
            #print(self.currVocT)
            for trigram in self.vocTrigrams:
                if trigram in set(self.currVocT):
                    TF = (self.currVocT.count(trigram))/len(self.currVocT)
                    TFIDF = TF * np.log(self.IDF[trigram])
                    vec.append(TFIDF)
                else:
                    vec.append(0)
            self.bowT.append(vec)
        return (self.bowM, self.bowB, self.bowT)



            
if __name__ == "__main__":
    c = Corpus('./data/imdb.csv')
    m,b,t = c.getBOW()
    print(len(m))
    print(len(b))
    print(len(t))
    print(m)





        
