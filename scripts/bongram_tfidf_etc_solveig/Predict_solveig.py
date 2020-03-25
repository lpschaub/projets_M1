from glob import glob

import re

import spacy
from spacy import displacy
# nlp = spacy.load('en')

class Voc(object) : 

    def __init__(self, corpus) :

        self.corpus = corpus
        self.voc = []

    def build_voc(self) : 
        for fic in self.corpus.lire() : 
            # print(fic)
            ficstring = self.clean(self.corpus.string(fic).replace('\n',' '))
            for elem in ficstring.split() : 
                if elem != "" : 
                    self.voc.append(elem)
        self.voc = list(set(self.voc))
        print(self.voc) # à vous de compléter
    

    def clean(self,ficstring) : 
        ficstring = ficstring.replace('(','')
        ficstring = ficstring.replace(')','')
        return ficstring


    def ecrire_voc(self,out) : 

        out.write('\n'.join([elem for elem in self.voc]))

    def load_voc(self, vocfile) : 

        self.voc = vocfile

class VocCSV(Voc) :

    def __init__(self, corpus) :
        Voc.__init__(self, corpus)
        self.bigram = self.build_ngram(2)
        self.trigram = self.build_ngram(3)
    
    def build_voc(self): 

        with open (self.corpus, 'r') as file:
            for line in file :
                comm = line.split('\t')[0]
                mots = comm.split()
                for elem in mots :
                    self.voc.append(elem)
        self.voc = list(set(self.voc))
        #print(self.voc)
        
    def build_ngram(self, n): 

        self.ngram = []
        with open (self.corpus, 'r') as file:
            for line in file : 
                comm = line.split('\t')[0]
                mots = comm.split()
                i = 0
                while i <= len(mots)-n:
                    self.ngram.append(" ".join(mots[i:i+n]))
                    i += 1
        self.ngram = list(set(self.ngram))
        return self.ngram


class Corpus(object) : 

    def __init__(self, path, voc = "") : 

        self.corpus = path
        if voc : 
            self.voc = voc
        else : 
            voc = Voc(self, self.corpus)
            voc.build_voc()
            self.voc = voc.voc
        self.bow = []

    def lire (self) :
        return glob(self.corpus+"/*")


    def string(self,fic) : 

        return open(fic).read()

    def getBOW(self) : 
        x = 0

        for fic in self.lire() : 

            self.bow.append([])
            ficstring = self.string(fic).replace('\n',' ')

            for elem in self.voc : 

                if elem in ficstring : 
                    self.bow[x].append(1)  
                else : 
                    self.bow[x].append(0)
            print("done")

            x += 1

        # print(self.bow)

class CorpusCSV(Corpus) : 

    def __init__(self, path, voc = "") : 
        self.corpus = path
        if voc : 
            self.voc = voc
        else : 
            voc = VocCSV(path)
            voc.build_voc()
            self.voc = voc.voc
            self.bigram = voc.bigram
            self.trigram = voc.trigram
        self.bow = []
        self.tf_idf=[]


    def lire(self): 
        return open(self.corpus).read()

    def getBOW(self) : 
        x = 0

        with open (self.corpus, 'r') as file:
            for line in file:

                if "text\t" in line:
                    continue

                self.bow.append([])
                comm = line.split('\t')[0]

                for elem in self.voc : 

                    if elem in comm : 
                        self.bow[x].append(1)  
                    else : 
                        self.bow[x].append(0)

                x += 1
            #print(x)
            
    def getBONgram(self, ngram) : 
        x = 0
        bongram = []

        with open (self.corpus, 'r') as file:
            for line in file:

                if "text\t" in line:
                    continue

                bongram.append([])
                comm = line.split('\t')[0]

                for elem in ngram : 

                    if elem in comm : 
                        bongram[x].append(1)  
                    else : 
                        bongram[x].append(0)

                x += 1
            #print(x)
        return bongram
        
    def getIDF(self, mot):
        nbl = 0
        nbl_mot = 0
        with open (self.corpus, "r") as file:
            for line in file:
                if "text\t" in line:
                    continue
                comm = line.split('\t')[0]
                if mot in comm:
                    nbl_mot += 1
                nbl += 1
        return nbl/nbl_mot
        
    def getTF_IDF(self):
        with open (self.corpus, "r") as file:
            x = 0
            for line in file:
                if "text\t" in line:
                    continue
                comm = line.split('\t')[0]
                mots = comm.split()
                self.tf_idf.append([])
                for word in self.voc:
                    freq = 0
                    for mot in mots:
                        if mot == word:
                            freq += 1
                    if freq == 0:
                        self.tf_idf[x].append(0)
                    else:
                        idf = self.getIDF(word)
                        self.tf_idf[x].append((freq/len(mots))*idf)
                x += 1

    def getTF_IDFngram(self, ngram, n):
        tf_idf_ngram = []
        with open (self.corpus, "r") as file:
            x = 0
            for line in file:
                if "text\t" in line:
                    continue
                comm = line.split('\t')[0]
                mots = comm.split()
                ngraml = []
                i = 0
                while i <= len(mots)-n:
                    ngraml.append(" ".join(mots[i:i+n]))
                    i += 1
                tf_idf_ngram.append([])
                for elem in ngram:
                    freq = 0
                    for item in ngraml:
                        if item == elem:
                            freq += 1
                    if freq == 0:
                        self.tf_idf[x].append(0)
                    else:
                        idf = self.getIDF(elem)
                        tf_idf_ngram[x].append((freq/len(mots))*idf)
                x += 1
        return tf_idf_ngram

class Predict(object) : 


    def __init__(self, doc, nlp, seuil = 0.5) : 

        self.doc = doc
        self.seuil = seuil
        self.docnettoye = re.sub(r"<[^>]+>", " ", self.doc)
        self.nlp = nlp(self.docnettoye)


    def workspace(self) : 

        self.tokens = []
        self.lemmas = []
        self.tag = []
        self.pos = []
        for token in self.nlp:
            if token.is_punct == False and token.is_space == False:
                self.tokens.append(token.text)
                self.lemmas.append(token.lemma_)
                self.tag.append(token.tag_)
                self.pos.append(token.pos_)

        for lemma in self.lemmas:
            n = self.lemmas.index(lemma)
            if lemma in ("interesting", "interested", "good", "recommended", "recommend" "excellent", "convincing", "thrilling", "satisfy", "amazing", "amazement", "beautiful", "delightful", "sublime", "great", "joyous", "fun", "funny", "legendary", "nice", "astounding", "enjoy", "fascinating", "fascinate", "remarkable", "memorable", "entertaining", "wonderful", "likable", "nifty", "favorite", "clever", "cleverly", "amusing", "gem", "chemistry", "masterpiece") or (lemma == "well" and n < len(self.lemmas)-1 and self.tag[n+1] == "VBN"):
                if "not" in (self.lemmas[n-3:n-1]) or "no" in (self.lemmas[n-3:n-1]) or "nothing" in (self.lemmas[n-3:n-1]) or "neither" in (self.lemmas[n-3:n-1]) or "nor" in (self.lemmas[n-3:n-1]) or "without" in (self.lemmas[n-3:n-1]) or "could" in (self.lemmas[n-5:n-2]) or "would" in (self.lemmas[n-5:n-2]) or "but" in (self.lemmas[n+1:n+4]) or "if" in (self.lemmas[n-5:n-2]):
                    self.seuil -= 0.1
                else:
                    self.seuil += 0.1
                    if "so" in (self.lemmas[n-3:n-1]) or "truly" in (self.lemmas[n-3:n-1]) or "really" in (self.lemmas[n-3:n-1]) or "very" in (self.lemmas[n-3:n-1]) or "completely" in (self.lemmas[n-3:n-1]) or "highly" in (self.lemmas[n-3:n-1]) or "deeply" in (self.lemmas[n-3:n-1]):
                        self.seuil += 0.1
                    if self.tokens[n] == self.tokens[n].upper():
                        self.seuil += 0.1

            elif (lemma in ("horrible", "lame", "despicable", "boring", "bored", "crap", "awful", "appalled", "scandalous", "sadly", "sad" "why", "how", "where" "failure", "fail", "ridiculous", "painful", "painfully", "horrendous", "disaster", "waste", "bad", "disappointed", "disappointment", "disappointing", "irritating" "pointless", "turgid", "emotionless", "embarrassed", "clichéd", "cliché" "stupid", "worthless", "bleak", "miscast", "weak", "problem", "incoherent", "unsuccessfull", "wooden", "annoying", "implausible", "overdone", "bizarre", "fake", "stereotype", "dangerous", "mediocrity", "mediocre", "dull", "terrible", "wrong", "empty", "silly", "poorly", "poor", "laughable", "pass", "hollow", "shallow", "message-less", "unfortunate", "inept", "unfunny", "deception", "pointlessly", "disastrous", "unnecessary", "unappealing", "goofy", "suck", "insane", "half-written", "pretentious", "unpleasant", "offensive", "feeble") and "not" not in (self.lemmas[n-3:n-1]) and "no" not in (self.lemmas[n-3:n-1]) and "nothing" not in (self.lemmas[n-3:n-1]) and "neither" not in (self.lemmas[n-3:n-1]) and "nor" not in (self.lemmas[n-3:n-1]) and "without" not in (self.lemmas[n-3:n-1])) or (lemma in ("plot", "development", "coherence") and ("without" in (self.lemmas[n-3:n-1]) or "no" in (self.lemmas[n-3:n-1])) or (lemma == "too" and n < len(self.lemmas)-1 and self.pos[n+1] == "ADJ")):
                self.seuil -= 0.1
                if "so" in (self.lemmas[n-3:n-1]) or "truly" in (self.lemmas[n-3:n-1]) or "really" in (self.lemmas[n-3:n-1]) or "very" in (self.lemmas[n-3:n-1]) or "completely" in (self.lemmas[n-3:n-1]) or "highly" in (self.lemmas[n-3:n-1]) or "deeply" in (self.lemmas[n-3:n-1]):
                    self.seuil -= 0.1
                if self.tokens[n] == self.tokens[n].upper():
                    self.seuil -= 0.1


        for token in self.tokens:
            n = self.tokens.index(token)
            if token == "worth" :
                try : 
                    if  self.tokens[n+1] == "watching" :
                        if "not" in (self.tokens[n-3:n-1]):
                            self.seuil  -= 0.5
                        else:
                            self.seuil  += 0.5
                except IndexError : 
                    print(self.tokens)
                    print(self.tokens[n])


        if "unfortunately" in self.lemmas or "alas" in self.lemmas or "I wanted to like" in self.doc or "doesn't save this film" in self.doc or "don't go" in self.doc or "screwed up" in self.doc or "avoid watching" in self.doc or "avoid this film" in self.doc or "don't see it" in self.doc:
            self.seuil  -= 0.5
        elif "director should" in self.doc or "if only" in self.doc or "own risk" in self.doc or "look elsewhere" in self.doc:
            self.seuil -= 0.2



    def predict(self) :

        self.workspace()

        if self.seuil < 0.5 : 
            self.predicted = 'neg'
        else : 
            self.predicted = 'pos'


if __name__ == '__main__':

    c = CorpusCSV('./imdb_reduit.tsv')
    c.getBOW()
    print(c.bow)
    #print(c.bigram)
    #print(c.trigram)
    bo2gram = c.getBONgram(c.bigram)
    print(bo2gram)
    bo3gram = c.getBONgram(c.trigram)
    print(bo3gram)
    
    c.getTF_IDF()
    print(c.tf_idf)
    
    tf_idf_2gram = c.getTF_IDFngram(c.bigram, 2)
    print(tf_idf_2gram)
    tf_idf_3gram = c.getTF_IDFngram(c.trigram, 3)
    print(tf_idf_3gram)

