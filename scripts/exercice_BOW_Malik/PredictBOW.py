import re
from glob import glob
import spacy
from spacy import displacy
nlp = spacy.load("en_core_web_sm")



class Voc(object):
    def __init__(self, corpus):
        self.corpus = corpus
        self.voc = []
    def build_voc(self):
        for fic in self.corpus.lire():
            print(fic)
            ficstring = self.clean(self.corpus.string(fic).replace('\n',' '))
            text = nlp(ficstring)
            #print(ficstring)
            for token in text:
                if(token.text.lower() not in self.voc):
                    self.voc.append(token.text.lower())
        return self.voc
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
            self.voc = voc.voc
        self.bow = []
    def lire(self):
        return glob(self.corpus+"/*")
    def string(self, fic):
        return open(fic).read()
    def getBOW(self):
        x = 0
        patt = re.compile('<[^>]+>')
        for fic in self.lire():
            currVoc = []
            currText = self.string(fic)
            currText = currText.replace('\n',' ')
            currText = currText.replace('(', '')
            currText = currText.replace(')', '')
            currText = re.sub(patt, ' ', currText)
            currText = re.sub('\.', r'. ', currText)
            text = nlp(currText)
            for token in text:
                if(token.text.lower() not in currVoc):
                    currVoc.append(token.text.lower())
            #print(currVoc)
            vec = []
            for word in self.voc:
                if word in currVoc:
                    vec.append(1)
                else:
                    vec.append(0)
            #print(vec)
            self.bow.append(vec)
        return self.bow

            

if __name__ == "__main__":
    c = Corpus('../corpus3/imdb/neg')
    b = c.getBOW()
    print(b)





        
