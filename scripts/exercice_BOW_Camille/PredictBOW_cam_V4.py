from glob import glob
from collections import Counter
import re
import nltk
import math
from nltk.util import ngrams
import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_sm')
# Utiliser la classe Corpus pour un corpus créé à partir de dossiers de fichiers/ corpusCsv pour créer a partir d'un dataframe pandas (fonctionnement quasi identique des 2 classes, juste 2 methodes different : string() et lire())
class Voc(object) :

	def __init__(self, corpus) :

		self.corpus = corpus
		self.unigram={}
		self.bigram={}
		self.trigram={}

	def build_voc_ngram(self,n,min=0.1,max=0.95):
		vocText=Counter()
		for text in self.corpus.corpus_norm :
			for ngram in [' '.join(ngram) for ngram in set(ngrams(text.split(),n))]:
				vocText[ngram]+=1
		longueur_corp=len(self.corpus.corpus_norm)
		return [(mot,(count/longueur_corp)) for mot,count in vocText.items() if count/longueur_corp>=min and count/longueur_corp<=max]

class Corpus(object) :

	def __init__(self, path, voc = "") :

		self.corpus = path
		self.corpus_norm=[]
		for fic in self.lire() :
			ficstring = self.clean(self.string(fic).replace('\n',' '))
			ligne=[item.lemma_ for item in nlp(ficstring) if  not item.is_space and not item.is_punct and not item.is_stop]
			self.corpus_norm.append(" "+' '.join(ligne)+" ")

		if voc :
			self.voc = voc
		else :
			self.voc = Voc(self)
			self.voc.unigram = self.voc.build_voc_ngram(1,0.1,0.95)
			self.voc.bigram=self.voc.build_voc_ngram(2,0.02,1)
			self.voc.trigram=self.voc.build_voc_ngram(3,0.02,1)

	def lire (self) :
		return glob(self.corpus+"/*")

	def clean(self,ficstring) :
		ficstring = re.sub('(<[^>]*>)|\(|\)',' ',ficstring)
		ficstring = re.sub(' {2,}',' ',ficstring)
		return ficstring

	def string(self,fic) :
		return open(fic).read()

	def getBOW(self, voc) :
		bow=[]
		for texte in self.corpus_norm :
			ligne=[]
			for mot in voc:
				regexp=re.compile(r' '+re.escape(mot[0])+r' ') #prendre en compte frontieres de mots pour pas de résultats trop faussés
				ligne.append(len(regexp.findall(texte)))
			bow.append(ligne)
		return bow

	def getBOW_unigram(self):
		self.bow_unigram=self.getBOW(self.voc.unigram)

	def getBOW_bigram(self):
		self.bow_bigram=self.getBOW(self.voc.bigram)

	def getBOW_trigram(self):
		self.bow_trigram=self.getBOW(self.voc.trigram)

	def getTFIDF(self,voc,ngram) :
		tfidf=[]
		for texte in self.corpus_norm :
			ligne=[]
			textList=texte.split()
			for mot in voc:
				regexp=re.compile(r' '+re.escape(mot[0])+r' ') # espace pour pas avoir un match pour "bit" avec le mot "bitter" par exemple (frontiere de mot pose pb pour les mots de types a.m. ) du coup il a fallu rajouter un espace en debut et fin  dans corpus norm
				tf=len(regexp.findall(texte))/(len(textList)+1-ngram) #+1-ngram : prendre en compte le fait qu'en fonction du ngram étudié, la longueur totale d'éléments n'est pas la même (pour 5 mots : 5 unigrams, 4 bigrams, 3 trigrams...
				idf=math.log2(1/mot[1])
				ligne.append(tf*idf)
			tfidf.append(ligne)
		return tfidf

	def getTFIDF_unigram(self):
			self.tfidf_unigram=self.getTFIDF(self.voc.unigram,1)

	def getTFIDF_bigram(self):
			self.tfidf_bigram=self.getTFIDF(self.voc.bigram,2)

	def getTFIDF_trigram(self):
			self.tfidf_trigram=self.getTFIDF(self.voc.trigram,3)

class CorpusCSV(Corpus) :

	def __init__(self, path, voc = "") :
		Corpus.__init__(self, path, voc)

	def string(self, fic) :
		return fic

	def lire(self):
		return self.corpus['text']

if __name__ == '__main__':

	c = Corpus('../../../imdb_test/neg')
	print(c.voc)
	c.getBOW()
	print(c.bow)
