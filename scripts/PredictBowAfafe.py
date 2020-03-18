from glob import glob
import re
import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_sm')


class Voc(object):
    def __init__(self, corpus):
        self.corpus = corpus
        self.voc = []

    def build_voc(self):
        vocab = {}
        i = 0
        for fic in self.corpus.lire():
            print(fic)
            ficstring = self.clean(self.corpus.string(fic).replace('\n', ' '))
            ficnlp = nlp(ficstring)
            for word in ficnlp:
                if word in vocab:
                    continue
                else:
                    vocab[word] = i
                    i += 1
            print(vocab)

    def clean(self, ficstring):
        ficstring = ficstring.replace('(', '')
        ficstring = ficstring.replace(')', '')
        ficstring = ficstring.replace('br', '')
        ficstring = ficstring.replace('<', '')
        ficstring = ficstring.replace('>', '')
        ficstring = ficstring.replace('/', '')
        return ficstring

    def ecrire_voc(self, out):
        out.write('\n'.join([elem for elem in self.voc]))

    def load_voc(self, vocfile):
        self.voc = Voc.string(vocfile)


class Corpus(object):
    def __init__(self, path, voc=""):
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
        for fic in self.lire():
            self.bow.append([])
            pass # 


if __name__ == '__main__':

	c = Corpus('../corpus/imdb/neg')
	c.getBOW()
