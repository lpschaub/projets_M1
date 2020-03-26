from glob import glob
import re
import nltk
import string
import csv
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
#import spacy
#from spacy import displacy
# nlp = spacy.load('en')

def clean(fic) : 

	ficclean = (fic.replace('\n',' ')).lower()
	liste = ['(',')','\\']
	for i in liste :
		ficclean = ficclean.replace(i,'')
	return ficclean
		
class Voc(object) : 

	def __init__(self, corpus) :

		self.corpus = corpus
		self.voc = []
		self.bi = []
		self.tri = []

	def build_voc(self) : 
		read = self.corpus.lire()
		next(read)
		for line in read : 
			voc2=[]
			lineclean = clean(line[0])
			fictok = word_tokenize(lineclean)
			for i in fictok :
				(voc2).append(i)
				if i not in self.voc and i not in string.punctuation:
					(self.voc).append(i)
			self.bi+=list(ngrams(voc2,2))
			self.tri+=list(ngrams(voc2,3))

	def ecrire_voc(self,out) : 

		out.write('\n'.join([elem for elem in self.voc]))


class CSVCorpus(object) : 

	def __init__(self, path, voc = "") : 

		self.corpus = path
		voc = Voc(self)
		voc.build_voc()
		self.voc = voc.voc
		self.bi = voc.bi
		self.tri = voc.tri
		self.bow = []
		self.bo_bi = []
		self.bo_tri = []
		self.tf_idf = {}
		self.mat_tfidf = []
	

	def lire (self) :
		return csv.reader((open(self.corpus, "r", encoding="utf-8")), delimiter='\t')


	def getBOW(self) : 
		read = self.lire()
		next(read)
		for line in read :
			is_in = []
			lineclean = clean(line[0])
			linetok = word_tokenize(lineclean)
			for word in self.voc : 
				if word in linetok :	 
					is_in.append(1)
				else :
					is_in.append(0)
			self.bow.append(is_in)
			
	def getBO_bi(self) : 
		read = self.lire()
		next(read)
		for line in read :
			is_in = []
			lineclean = clean(line[0])
			linetok = word_tokenize(lineclean)
			line_bi = list(ngrams(linetok,2))
			for bi in self.bi : 
				if bi in line_bi :	 
					is_in.append(1)
				else :
					is_in.append(0)
			self.bo_bi.append(is_in)
			
	def getBO_tri(self) : 
		read = self.lire()
		next(read)
		for line in read :
			is_in = []
			lineclean = clean(line[0])
			linetok = word_tokenize(lineclean)
			line_tri = list(ngrams(linetok,3))
			for tri in self.tri : 
				if tri in line_tri :	 
					is_in.append(1)
				else :
					is_in.append(0)
			self.bo_tri.append(is_in)
			
	def getTF_IDF(self) :
		nbl=-1
		for line in self.lire() :
			nbl+=1
		num_line = 0
		read = self.lire()
		next(read)
		for line in read :
			num_line +=1
			cntr = Counter()
			lineclean = clean(line[0])
			linetok = word_tokenize(lineclean)
			for word in linetok :
				cntr[word]+=1
				
				nbl_word = 0
				read2=self.lire()
				next(read2)
				for line2 in read2 :
					lineclean2 = clean(line2[0])
					linetok2 = word_tokenize(lineclean2)
					if word in linetok2 :
						nbl_word +=1		   
					
			for word in linetok : 
				tf = cntr[word]/len(linetok)
				idf = nbl/nbl_word
				tfidf = tf*idf
				self.tf_idf[word,num_line]=tfidf
		  
	def getMat_Tfidf(self) :
		nbl = 0
		read = self.lire()
		next(read)
		for line in read :
			nbl +=1
			is_in = []
			lineclean = clean(line[0])
			linetok = word_tokenize(lineclean)
			for word in self.voc : 
				if word in linetok :	 
					is_in.append(self.tf_idf[word,nbl])
				else :
					is_in.append(0)
			self.mat_tfidf.append(is_in)
				


# class Predict(object) : 


#	def __init__(self, doc, nlp, seuil = 0.5) : 

#		self.doc = doc
#		self.seuil = seuil
#		self.docnettoye = re.sub(r"<[^>]+>", " ", self.doc)
#		self.nlp = nlp(self.docnettoye)


#	def workspace(self) : 

#		self.tokens = []
#		self.lemmas = []
#		self.tag = []
#		self.pos = []
#		for token in self.nlp:
#			if token.is_punct == False and token.is_space == False:
#				self.tokens.append(token.text)
#				self.lemmas.append(token.lemma_)
#				self.tag.append(token.tag_)
#				self.pos.append(token.pos_)

#		for lemma in self.lemmas:
#			n = self.lemmas.index(lemma)
#			if lemma in ("interesting", "interested", "good", "recommended", "recommend" "excellent", "convincing", "thrilling", "satisfy", "amazing", "amazement", "beautiful", "delightful", "sublime", "great", "joyous", "fun", "funny", "legendary", "nice", "astounding", "enjoy", "fascinating", "fascinate", "remarkable", "memorable", "entertaining", "wonderful", "likable", "nifty", "favorite", "clever", "cleverly", "amusing", "gem", "chemistry", "masterpiece") or (lemma == "well" and n < len(self.lemmas)-1 and self.tag[n+1] == "VBN"):
#				if "not" in (self.lemmas[n-3:n-1]) or "no" in (self.lemmas[n-3:n-1]) or "nothing" in (self.lemmas[n-3:n-1]) or "neither" in (self.lemmas[n-3:n-1]) or "nor" in (self.lemmas[n-3:n-1]) or "without" in (self.lemmas[n-3:n-1]) or "could" in (self.lemmas[n-5:n-2]) or "would" in (self.lemmas[n-5:n-2]) or "but" in (self.lemmas[n+1:n+4]) or "if" in (self.lemmas[n-5:n-2]):
#					self.seuil -= 0.1
#				else:
#					self.seuil += 0.1
#					if "so" in (self.lemmas[n-3:n-1]) or "truly" in (self.lemmas[n-3:n-1]) or "really" in (self.lemmas[n-3:n-1]) or "very" in (self.lemmas[n-3:n-1]) or "completely" in (self.lemmas[n-3:n-1]) or "highly" in (self.lemmas[n-3:n-1]) or "deeply" in (self.lemmas[n-3:n-1]):
#						self.seuil += 0.1
#					if self.tokens[n] == self.tokens[n].upper():
#						self.seuil += 0.1

#			elif (lemma in ("horrible", "lame", "despicable", "boring", "bored", "crap", "awful", "appalled", "scandalous", "sadly", "sad" "why", "how", "where" "failure", "fail", "ridiculous", "painful", "painfully", "horrendous", "disaster", "waste", "bad", "disappointed", "disappointment", "disappointing", "irritating" "pointless", "turgid", "emotionless", "embarrassed", "clichéd", "cliché" "stupid", "worthless", "bleak", "miscast", "weak", "problem", "incoherent", "unsuccessfull", "wooden", "annoying", "implausible", "overdone", "bizarre", "fake", "stereotype", "dangerous", "mediocrity", "mediocre", "dull", "terrible", "wrong", "empty", "silly", "poorly", "poor", "laughable", "pass", "hollow", "shallow", "message-less", "unfortunate", "inept", "unfunny", "deception", "pointlessly", "disastrous", "unnecessary", "unappealing", "goofy", "suck", "insane", "half-written", "pretentious", "unpleasant", "offensive", "feeble") and "not" not in (self.lemmas[n-3:n-1]) and "no" not in (self.lemmas[n-3:n-1]) and "nothing" not in (self.lemmas[n-3:n-1]) and "neither" not in (self.lemmas[n-3:n-1]) and "nor" not in (self.lemmas[n-3:n-1]) and "without" not in (self.lemmas[n-3:n-1])) or (lemma in ("plot", "development", "coherence") and ("without" in (self.lemmas[n-3:n-1]) or "no" in (self.lemmas[n-3:n-1])) or (lemma == "too" and n < len(self.lemmas)-1 and self.pos[n+1] == "ADJ")):
#				self.seuil -= 0.1
#				if "so" in (self.lemmas[n-3:n-1]) or "truly" in (self.lemmas[n-3:n-1]) or "really" in (self.lemmas[n-3:n-1]) or "very" in (self.lemmas[n-3:n-1]) or "completely" in (self.lemmas[n-3:n-1]) or "highly" in (self.lemmas[n-3:n-1]) or "deeply" in (self.lemmas[n-3:n-1]):
#					self.seuil -= 0.1
#				if self.tokens[n] == self.tokens[n].upper():
#					self.seuil -= 0.1


#		for token in self.tokens:
#			n = self.tokens.index(token)
#			if token == "worth" :
#				try : 
#					if	self.tokens[n+1] == "watching" :
#						if "not" in (self.tokens[n-3:n-1]):
#							self.seuil	-= 0.5
#						else:
#							self.seuil	+= 0.5
#				except IndexError : 
#					print(self.tokens)
#					print(self.tokens[n])


#		if "unfortunately" in self.lemmas or "alas" in self.lemmas or "I wanted to like" in self.doc or "doesn't save this film" in self.doc or "don't go" in self.doc or "screwed up" in self.doc or "avoid watching" in self.doc or "avoid this film" in self.doc or "don't see it" in self.doc:
#			self.seuil	-= 0.5
#		elif "director should" in self.doc or "if only" in self.doc or "own risk" in self.doc or "look elsewhere" in self.doc:
#			self.seuil -= 0.2



#	def predict(self) :

#		self.workspace()

#		if self.seuil < 0.5 : 
#			self.predicted = 'neg'
#		else : 
#			self.predicted = 'pos'


if __name__ == '__main__':

	c = CSVCorpus('avis.tsv')
	vocab = Voc(c)
	vocab.build_voc()
	print(vocab.voc)
	c.getBOW()
	print(c.bow)
	print(c.bi)
	c.getBO_bi()
	print(c.bo_bi)
	print(c.tri)
	c.getBO_tri()
	print(c.bo_tri)
	c.getTF_IDF()
	c.getMat_Tfidf()
	print(c.mat_tfidf)