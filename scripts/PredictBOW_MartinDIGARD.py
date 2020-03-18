# Martin Digard

from glob import glob

import re

import spacy
from spacy import displacy
nlp = spacy.load('en')


class Voc(object):

    def __init__(self, corpus):

        self.corpus = corpus
        self.voc = []

    def build_voc(self):
        for fic in self.corpus.lire():
            ficstring = self.clean(self.corpus.string(fic).replace('\n', ' '))
            tokens = nlp(ficstring)
            for token in tokens:
                if not (token.is_punct or token.is_space):
                    if str(token) not in self.voc:
                        self.voc.append(str(token))


    def clean(self, ficstring):
        ficstring = ficstring.replace('(', '')
        ficstring = ficstring.replace(')', '')
        ficstring = re.sub("<br|/>", '', ficstring)
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
        return sorted(glob(self.corpus+"/*"))

    def string(self, fic):

        return open(fic).read()

    def getBOW(self):
        x = 0
        for fic in self.lire():
            doc = []
            with open(fic) as f:
                f = f.read()
                for mot in self.voc:
                    if mot in f:
                        doc.append('1')
                    else:
                        doc.append('0')
            self.bow.append(doc)
        for e in self.bow:
            print(e)


if __name__ == '__main__':

    c = Corpus('corpus/imdb/*')
    # c = Corpus('corpus/test_mini_1')
    c.getBOW()
