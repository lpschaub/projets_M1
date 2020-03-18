from glob import glob

import re, csv
import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_sm')

class Voc(object) : 

	def __init__(self, corpus) :
		
		self.corpus = corpus
		self.voc = []
		self.vocfile = "vocfile.txt"

	def build_voc(self) : 
		for fic in self.corpus.lire() : 
			print(fic)		
			wordslist = self.getfilewordlist(fic)		
			for word in wordslist :
				if word not in self.voc and word != "\'" and word != " " :
					self.voc.append(word)
		
		with open(self.vocfile, 'w') as out :
			self.ecrire_voc(out)
			
	def getfilewordlist(self, fic) :
		wordlist = []
		with open(fic, 'r') as ficstring :
			ficstring = ficstring.read()
			ficstring = Voc.clean(self, ficstring)
			for word in nlp(ficstring) :
				wordlist.append(str(word).lower())
		return wordlist
		

	def clean(self,ficstring) : 
		exception = [".", ";", ":", ",", "!", "?", "(", ")", ">", "<", "\"", "...", "/", "-", "_", "\t", ""]
		ficstring = re.sub("<br|/>", '', ficstring)
		for item in exception :
			ficstring = ficstring.replace(item,'')
		ficstring = ficstring.replace('\n',' ')
		return ficstring


	def ecrire_voc(self,out) : 

		out.write('\n'.join([elem for elem in self.voc]))
		print(f"Vocabulary file created in current directory.")

	def load_voc(self, vocfile) : 

		self.voc = Voc.string(vocfile)


class Corpus(object) : 

	def __init__(self, path, voc = "") : 
        
		self.corpus = path
		#Que vérifie voc ?
		if voc : 
			self.voc = Voc.load_voc(voc)
			print("Voc loaded")
		else : 
			voc = Voc(self)
			voc.build_voc()
			print("Voc created.")
			self.voc = voc.voc
		self.bow = []

	def lire(self) :
		return glob(self.corpus+"/*")


	def string(self,fic) : 

		return open(fic).read()

	def getBOW(self) : 
		x = 0
		for fic in self.lire() :
			if x == 500:
				break
			filebow = []
			wordlist = Voc.getfilewordlist(self, fic)
			for item in self.voc :
				if item in wordlist :
					filebow.append('1')
				else :
					filebow.append('0')
			self.bow.append(filebow)
			x+=1
		self.bowcsv(self.bow)
		print("Bag of words saved as .csv in current directory.") 
				
	def bowcsv(self, bow) :
		with open('bagofwords.tsv', 'w', newline='\n', encoding='utf-8') as file :
			writer = csv.writer(file, delimiter= '\t')
			writer.writerows(self.bow)


# class Predict(object) : 


# 	def __init__(self, doc, nlp, seuil = 0.5) : 

# 		self.doc = doc
# 		self.seuil = seuil
# 		self.docnettoye = re.sub(r"<[^>]+>", " ", self.doc)
# 		self.nlp = nlp(self.docnettoye)


# 	def workspace(self) : 

# 		self.tokens = []
# 		self.lemmas = []
# 		self.tag = []
# 		self.pos = []
# 		for token in self.nlp:
# 			if token.is_punct == False and token.is_space == False:
# 				self.tokens.append(token.text)
# 				self.lemmas.append(token.lemma_)
# 				self.tag.append(token.tag_)
# 				self.pos.append(token.pos_)

# 		for lemma in self.lemmas:
# 			n = self.lemmas.index(lemma)
# 			if lemma in ("interesting", "interested", "good", "recommended", "recommend" "excellent", "convincing", "thrilling", "satisfy", "amazing", "amazement", "beautiful", "delightful", "sublime", "great", "joyous", "fun", "funny", "legendary", "nice", "astounding", "enjoy", "fascinating", "fascinate", "remarkable", "memorable", "entertaining", "wonderful", "likable", "nifty", "favorite", "clever", "cleverly", "amusing", "gem", "chemistry", "masterpiece") or (lemma == "well" and n < len(self.lemmas)-1 and self.tag[n+1] == "VBN"):
# 				if "not" in (self.lemmas[n-3:n-1]) or "no" in (self.lemmas[n-3:n-1]) or "nothing" in (self.lemmas[n-3:n-1]) or "neither" in (self.lemmas[n-3:n-1]) or "nor" in (self.lemmas[n-3:n-1]) or "without" in (self.lemmas[n-3:n-1]) or "could" in (self.lemmas[n-5:n-2]) or "would" in (self.lemmas[n-5:n-2]) or "but" in (self.lemmas[n+1:n+4]) or "if" in (self.lemmas[n-5:n-2]):
# 					self.seuil -= 0.1
# 				else:
# 					self.seuil += 0.1
# 					if "so" in (self.lemmas[n-3:n-1]) or "truly" in (self.lemmas[n-3:n-1]) or "really" in (self.lemmas[n-3:n-1]) or "very" in (self.lemmas[n-3:n-1]) or "completely" in (self.lemmas[n-3:n-1]) or "highly" in (self.lemmas[n-3:n-1]) or "deeply" in (self.lemmas[n-3:n-1]):
# 						self.seuil += 0.1
# 					if self.tokens[n] == self.tokens[n].upper():
# 						self.seuil += 0.1

# 			elif (lemma in ("horrible", "lame", "despicable", "boring", "bored", "crap", "awful", "appalled", "scandalous", "sadly", "sad" "why", "how", "where" "failure", "fail", "ridiculous", "painful", "painfully", "horrendous", "disaster", "waste", "bad", "disappointed", "disappointment", "disappointing", "irritating" "pointless", "turgid", "emotionless", "embarrassed", "clichéd", "cliché" "stupid", "worthless", "bleak", "miscast", "weak", "problem", "incoherent", "unsuccessfull", "wooden", "annoying", "implausible", "overdone", "bizarre", "fake", "stereotype", "dangerous", "mediocrity", "mediocre", "dull", "terrible", "wrong", "empty", "silly", "poorly", "poor", "laughable", "pass", "hollow", "shallow", "message-less", "unfortunate", "inept", "unfunny", "deception", "pointlessly", "disastrous", "unnecessary", "unappealing", "goofy", "suck", "insane", "half-written", "pretentious", "unpleasant", "offensive", "feeble") and "not" not in (self.lemmas[n-3:n-1]) and "no" not in (self.lemmas[n-3:n-1]) and "nothing" not in (self.lemmas[n-3:n-1]) and "neither" not in (self.lemmas[n-3:n-1]) and "nor" not in (self.lemmas[n-3:n-1]) and "without" not in (self.lemmas[n-3:n-1])) or (lemma in ("plot", "development", "coherence") and ("without" in (self.lemmas[n-3:n-1]) or "no" in (self.lemmas[n-3:n-1])) or (lemma == "too" and n < len(self.lemmas)-1 and self.pos[n+1] == "ADJ")):
# 				self.seuil -= 0.1
# 				if "so" in (self.lemmas[n-3:n-1]) or "truly" in (self.lemmas[n-3:n-1]) or "really" in (self.lemmas[n-3:n-1]) or "very" in (self.lemmas[n-3:n-1]) or "completely" in (self.lemmas[n-3:n-1]) or "highly" in (self.lemmas[n-3:n-1]) or "deeply" in (self.lemmas[n-3:n-1]):
# 					self.seuil -= 0.1
# 				if self.tokens[n] == self.tokens[n].upper():
# 					self.seuil -= 0.1


# 		for token in self.tokens:
# 			n = self.tokens.index(token)
# 			if token == "worth" :
# 				try : 
# 					if  self.tokens[n+1] == "watching" :
# 						if "not" in (self.tokens[n-3:n-1]):
# 							self.seuil  -= 0.5
# 						else:
# 							self.seuil  += 0.5
# 				except IndexError : 
# 					print(self.tokens)
# 					print(self.tokens[n])


# 		if "unfortunately" in self.lemmas or "alas" in self.lemmas or "I wanted to like" in self.doc or "doesn't save this film" in self.doc or "don't go" in self.doc or "screwed up" in self.doc or "avoid watching" in self.doc or "avoid this film" in self.doc or "don't see it" in self.doc:
# 			self.seuil  -= 0.5
# 		elif "director should" in self.doc or "if only" in self.doc or "own risk" in self.doc or "look elsewhere" in self.doc:
# 			self.seuil -= 0.2



# 	def predict(self) :

# 		self.workspace()

# 		if self.seuil < 0.5 : 
# 			self.predicted = 'neg'
# 		else : 
# 			self.predicted = 'pos'


if __name__ == '__main__':
	
	
	c = Corpus('../corpus/imdb/*')
	c.getBOW()
