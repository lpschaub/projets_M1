from glob import glob

import re

import spacy
from spacy import displacy
nlp = spacy.load('en')

class Predict(object) : 


	def __init__(self, doc, seuil = 0.5) : 

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
			if lemma in ("interesting", "interested", "good", "excellent", "convincing", "satisfy", "joyous", "fun", "funny", "nice", "enjoy", "entertaining", "likable", "nifty", "clever", "cleverly", "amusing", "chemistry", "cute") or (lemma == "well" and n < len(self.lemmas)-1 and self.tag[n+1] == "VBN"):
				if "not" in (self.lemmas[n-3:n-1]) or "no" in (self.lemmas[n-3:n-1]) or "nothing" in (self.lemmas[n-3:n-1]) or "neither" in (self.lemmas[n-3:n-1]) or "nor" in (self.lemmas[n-3:n-1]) or "without" in (self.lemmas[n-3:n-1]) or "could" in (self.lemmas[n-5:n-2]) or "would" in (self.lemmas[n-5:n-2]) or "but" in (self.lemmas[n+1:n+4]) or "if" in (self.lemmas[n-5:n-2]):
					self.seuil -= 0.1
				else:
					self.seuil += 0.1
					if "so" in (self.lemmas[n-3:n-1]) or "truly" in (self.lemmas[n-3:n-1]) or "really" in (self.lemmas[n-3:n-1]) or "very" in (self.lemmas[n-3:n-1]) or "completely" in (self.lemmas[n-3:n-1]) or "highly" in (self.lemmas[n-3:n-1]) or "deeply" in (self.lemmas[n-3:n-1]):
						self.seuil += 0.1
					if self.tokens[n] == self.tokens[n].upper():
						self.seuil += 0.1

		for lemma in self.lemmas:
			n = self.lemmas.index(lemma)
			if lemma in ("recommended", "recommend" "excellent", "thrilling", "amazing", "amazement", "beautiful", "delightful", "sublime", "great", "legendary", "astounding", "fascinating", "fascinate", "remarkable", "memorable", "wonderful", "favorite", "gem", "masterpiece", "love") or (lemma == "well" and n < len(self.lemmas)-1 and self.tag[n+1] == "VBN"):
				if "not" in (self.lemmas[n-3:n-1]) or "no" in (self.lemmas[n-3:n-1]) or "nothing" in (self.lemmas[n-3:n-1]) or "neither" in (self.lemmas[n-3:n-1]) or "nor" in (self.lemmas[n-3:n-1]) or "without" in (self.lemmas[n-3:n-1]) or "could" in (self.lemmas[n-5:n-2]) or "would" in (self.lemmas[n-5:n-2]) or "but" in (self.lemmas[n+1:n+4]) or "if" in (self.lemmas[n-5:n-2]):
					self.seuil -= 0.1
				else:
					self.seuil += 0.2
					if "so" in (self.lemmas[n-3:n-1]) or "truly" in (self.lemmas[n-3:n-1]) or "really" in (self.lemmas[n-3:n-1]) or "very" in (self.lemmas[n-3:n-1]) or "completely" in (self.lemmas[n-3:n-1]) or "highly" in (self.lemmas[n-3:n-1]) or "deeply" in (self.lemmas[n-3:n-1]):
						self.seuil += 0.1
					if self.tokens[n] == self.tokens[n].upper():
						self.seuil += 0.1

			elif lemma in ("sadly", "sad", "why", "how", "where", "painful", "painfully", "turgid", "emotionless", "embarrassed", "clichéd", "cliché", "bleak", "miscast", "weak", "overdone", "bizarre", "fake", "stereotype", "dangerous", "dull", "wrong", "poorly", "poor", "pass", "hollow", "shallow", "message-less", "unfortunate", "inept", "unfunny", "deception", "pointlessly", "unnecessary", "unappealing", "goofy", "insane", "half-written", "unpleasant", "offensive", "feeble", "grotesque", "television"):
				if"not" not in (self.lemmas[n-3:n-1]) and "no" not in (self.lemmas[n-3:n-1]) and "nothing" not in (self.lemmas[n-3:n-1]) and "neither" not in (self.lemmas[n-3:n-1]) and "nor" not in (self.lemmas[n-3:n-1]) and "without" not in (self.lemmas[n-3:n-1]):
					self.seuil -= 0.1
					if "so" in (self.lemmas[n-3:n-1]) or "truly" in (self.lemmas[n-3:n-1]) or "really" in (self.lemmas[n-3:n-1]) or "very" in (self.lemmas[n-3:n-1]) or "completely" in (self.lemmas[n-3:n-1]) or "highly" in (self.lemmas[n-3:n-1]) or "deeply" in (self.lemmas[n-3:n-1]):
						self.seuil -= 0.1
					if self.tokens[n] == self.tokens[n].upper():
						self.seuil -= 0.1

			elif lemma in ("horrible", "lame", "despicable", "boring", "bored", "crap", "awful", "appalled", "scandalous", "failure", "fail", "ridiculous", "painful", "painfully", "horrendous", "disaster", "waste", "bad", "disappointed", "disappointment", "disappointing", "irritating", "pointless", "stupid", "worthless", "problem", "incoherent", "unsuccessfull", "wooden", "annoying", "implausible", "mediocrity", "mediocre", "terrible", "empty", "silly", "laughable", "disastrous", "suck", "pretentious"):
				if"not" not in (self.lemmas[n-3:n-1]) and "no" not in (self.lemmas[n-3:n-1]) and "nothing" not in (self.lemmas[n-3:n-1]) and "neither" not in (self.lemmas[n-3:n-1]) and "nor" not in (self.lemmas[n-3:n-1]) and "without" not in (self.lemmas[n-3:n-1]):
					self.seuil -= 0.3
					if "so" in (self.lemmas[n-3:n-1]) or "truly" in (self.lemmas[n-3:n-1]) or "really" in (self.lemmas[n-3:n-1]) or "very" in (self.lemmas[n-3:n-1]) or "completely" in (self.lemmas[n-3:n-1]) or "highly" in (self.lemmas[n-3:n-1]) or "deeply" in (self.lemmas[n-3:n-1]):
						self.seuil -= 0.1
					if self.tokens[n] == self.tokens[n].upper():
						self.seuil -= 0.1

			elif (lemma in ("plot", "development", "coherence") and ("without" in (self.lemmas[n-3:n-1]) or "no" in (self.lemmas[n-3:n-1])) or (lemma == "too" and n < len(self.lemmas)-1 and self.pos[n+1] == "ADJ")):
				self.seuil -= 0.1

		for token in self.tokens:
			n = self.tokens.index(token)
			if token == "worth" and n < len(self.lemmas)-1 and self.tokens[n+1] == "watching" :
				if "not" in (self.tokens[n-3:n-1]):
					self.seuil  -= 0.5
				else:
					self.seuil  += 0.5


		if "unfortunately" in self.lemmas or "alas" in self.lemmas or "I wanted to like" in self.doc or "doesn't save this film" in self.doc or "don't go" in self.doc or "screwed up" in self.doc or "avoid watching" in self.doc or "avoid this film" in self.doc or "don't see it" in self.doc:
			self.seuil  -= 0.5
		elif "director should" in self.doc or "if only" in self.doc or "own risk" in self.doc or "look elsewhere" in self.doc or "low budget" in self.doc:
			self.seuil -= 0.2



	def predict(self) :

		self.workspace()

		if self.seuil < 0.4 : 
			self.predicted = 'neg'
		else : 
			self.predicted = 'pos'


if __name__ == '__main__':


	for fic in glob('./imdb/train/neg/*.txt'):
		pred = Predict(open(fic).read())
		pred.predict()
		if pred.predicted == "pos":
			print(fic)
		#print(pred.predicted)