from glob import glob
import re
import spacy
from spacy import displacy

class Voc(object):
    def __init__(self, corpus):
        self.corpus = corpus
        self.voc = []
        self.voc_bigrams = []
        self.voc_trigrams = []
        self.tfidf = []
        
    def build_voc(self):
        self.voc = []
        self.voc_bigrams = []
        self.voc_trigrams = []
        
        for fic in self.corpus.lire():
            ficstring = self.clean(self.corpus.string(fic).replace('\n',' '))
            ficstring = ficstring.lower()
            
            #CONSTRUCTION DU VOC
            for mot in ficstring.split():
                if mot[0].isalpha(): #On vérifie que il y a au moins un caractère dans le mot
                    if mot not in self.voc:
                        self.voc.append(mot)
            
            #CONSTRUCTION DU VOC DE BIGRAMMES
            liste_mots = ficstring.split()
            for x in range(0,len(liste_mots)-2):
                if liste_mots[x][0].isalpha() and liste_mots[x+1][0].isalpha():
                    bigram = liste_mots[x] + " " + liste_mots[x+1]
                    if bigram not in self.voc_bigrams:
                        self.voc_bigrams.append(bigram)
            
            #CONSTRUCTION DU VOC DE TRIGRAMMES
            for x in range(0,len(liste_mots)-3):
                if liste_mots[x][0].isalpha() and liste_mots[x+1][0].isalpha() and liste_mots[x+2][0].isalpha():
                    trigram = liste_mots[x] + " " + liste_mots[x+1] + " " + liste_mots[x+2]
                    if trigram not in self.voc_trigrams:
                        self.voc_trigrams.append(trigram)
                        
        self.voc = list(set(self.voc))
        print(f"\nVOCABULAIRE CORPUS :\n{self.voc}")
        
        self.voc_bigrams=list(set(self.voc_bigrams))
        print(f"\nVOCABULAIRE DE BIGRAMMES :\n{self.voc_bigrams}")
        
        self.voc_trigrams=list(set(self.voc_trigrams))
        print(f"\nVOCABULAIRE DE TRIGRAMMES :\n{self.voc_trigrams}")
        
#        return self.voc
        
    def clean(self,ficstring) :
        ficstring = re.sub('\(|\)|\"|,|\.|:|;|!|\?|/|\\|<i>|<hr>|<|>',' ',ficstring)
        ficstring = re.sub(' {2,}',' ',ficstring)

        return ficstring
    
    def ecrire_voc(self, out):
        out.write('\n'.join([elem for elem in self.voc]))
        
    def load_voc(self, vocfile):
        self.voc = vocfile
        
class VocCSV(Voc):
    def __init__(self, corpus):
        Voc.__init__(self, corpus)
    
#    def build_voc(self):
#        for lines in self.corpus:
#            #lines = lines.clean
#            lines = lines.replace('\n',' ')
#            for elem in lines.split():
#                self.voc.append(elem)
#        self.voc = list(set(self.voc))
#        print(f"\nVOCABULAIRE :\n{self.voc}")

class Corpus(object):
    def __init__(self, path, vocab=""):
        self.corpus = path
        if vocab:
            self.voc=vocab
        else:
            voc = Voc(self)
            voc.build_voc()
            self.voc=voc.voc
            self.voc_bigrams=voc.voc_bigrams
            self.voc_trigrams=voc.voc_trigrams
        self.bow = []
        self.bow_bigrams = []
        self.bow_trigrams = []
        
    def lire(self):
        return glob(self.corpus+"/*")
    
    def string(self, fic):
        return open(fic).read()
    
    def getBOW(self):
        x=0
        for fic in self.lire():
            self.bow.append([])
            self.bow_bigrams.append([])
            self.bow_trigrams.append([])
            
            contenu = self.string(fic).replace('\n',' ')
            contenu = re.sub('\(|\)|\"|,|\.|:|;|!|\?|/|\\|<i>|<hr>|<|>',' ',contenu)
            contenu = re.sub(' {2,}',' ',contenu)
            
            #contenu = contenu.lower()
            
            #BOW VOC
            for word in self.voc:
                if word in contenu:
                    self.bow[x].append(1)
                else:
                    self.bow[x].append(0)
                    
            #BOW VOC BIGRAMMES
            for bigram in self.voc_bigrams:
                if bigram in contenu:
                    self.bow_bigrams[x].append(1)
                else:
                    self.bow_bigrams[x].append(0)
                    
            #BOW VOC TRIGRAMMES
            for trigram in self.voc_trigrams:
                if trigram in contenu:
                    self.bow_trigrams[x].append(1)
                else:
                    self.bow_trigrams[x].append(0)

            x+=1
            
        print(f"\nBAG OF WORDS :\n{self.bow}")
        print(f"\nBAG OF BIGRAMS :\n{self.bow_bigrams}")
        print(f"\nBAG OF TRIGRAMS :\n{self.bow_trigrams}")
        
class CorpusCSV(Corpus):
    def __init__(self, path, vocab=""):
        self.corpus = path
        if vocab:
            self.voc = vocab
        else:
            voc = VocCSV(self)
            voc.build_voc()
            self.voc = voc.voc
            self.voc_bigrams=voc.voc_bigrams
            self.voc_trigrams=voc.voc_trigrams
        self.bow = []
        self.bow_bigrams = []
        self.bow_trigrams = []

    def string(self, fic):
        return fic
    
    def lire(self):
        return self.corpus['text']

if __name__ == '__main__':
    print("Lancement de predict")
    #c = Corpus('../corpus/imdb/*')
    #c = CorpusCSV('commentaires.csv')
    #c.getBOW()
