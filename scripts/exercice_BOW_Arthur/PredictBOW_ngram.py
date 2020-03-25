from glob import glob

import re, csv, sys, math
import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_sm')
vocfile = "vocfile.txt"
bigramvocfile = "bigramvocfile.txt"
trigramvocfile = "trigramvocfile.txt"
#nombre de documents qui seront traités 
nombredoc = 20

class Voc(object) : 

	def __init__(self, corpus) :
		
		self.corpus = corpus
		self.voc = []
		
	def build_voc(self) : 

		x=0
		for fic in self.corpus.lire() : 
			if x == nombredoc :
				break
			print(fic)
			wordslist = self.getfilewordlist(fic)
			if choice == 'bigram' :
				wordslist = self.getbigramlist(wordslist)
			if choice == 'trigram' :
				wordslist = self.gettrigramlist(wordslist)
			for word in wordslist :
				if word not in self.voc :
					self.voc.append(word)
					
		
			x+=1
			
		if choice == 'word' :
			self.ecrire_voc(vocfile)
		if choice == 'bigram' :
			self.ecrire_voc(bigramvocfile)
		if choice == 'trigram' :
			self.ecrire_voc(trigramvocfile)
	
		if dotfidf == 1 :
			pass
			
			
	def getfilewordlist(self, fic) :
		wordlist = []
		with open(fic, 'r') as ficstring :
			ficstring = ficstring.read()
			ficstring = Voc.clean(self, ficstring)
			for word in nlp(ficstring) :
				wordlist.append(str(word).lower())
			i = 0	
			for item in wordlist :
				if wordlist[i] == '\'' or wordlist[i] == " "  :
					wordlist.pop(i)
				i+=1
				if i == len(wordlist)+1 :
					break
		return wordlist
	
	def getbigramlist(self, wordslist) :
		bigramlist = []
		i = 0
		for item in wordslist : 
			bigram = item + " " + wordslist[i+1]
			bigramlist.append(bigram)
			i += 1
			if i+1 == len(wordslist) :
				break
		return bigramlist
		
	def gettrigramlist(self, wordslist) :
		trigramlist = []
		i = 0
		for item in wordslist : 
			trigram = item + " " + wordslist[i+1] + " " + wordslist[i+2]
			trigramlist.append(trigram)
			i += 1
			if i+2 == len(wordslist) :
				break
		return trigramlist
	
	
	def get_tf(self, wordlist) :
		tfwordslist = []
		for item in wordlist :
			nbr=0
			for i in range(0, len(wordlist)) :
				if item == wordlist[i] :
					nbr += 1
			tfwordlist.append((item, (nbr/len(wordlist))))
		print(tfwordlist)
		return tfwordlist
				

	def get_idf(self, wordlist) :
		pass
	
	def get_tfidf(self, tf, idf) :
		tf_idf = round((item[1] / document[1])*(math.log2(len(sentences)*(docnumber))),3)
	
	def clean(self,ficstring) : 
		exception = [".", ";", ":", ",", "!", "?", "(", ")", ">", "<", "\"", "...", "/", "-", "_", "\t", ""]
		ficstring = re.sub("<br|/>", '', ficstring)
		for item in exception :
			ficstring = ficstring.replace(item,'')
		ficstring = ficstring.replace('\n',' ')
		ficstring = ficstring.replace(' \' ',' ')
		ficstring = ficstring.replace("  ",' ')
		ficstring = ficstring.replace("   ",' ')
		ficstring = ficstring.replace("    ",' ')
		return ficstring


	def ecrire_voc(self, filename) : 
		with open(filename, 'w', encoding='utf-8') as out :
			out.write('\n'.join([elem for elem in self.voc]))
		print(f"Vocabulary file \'{filename}\' created in current directory.")

	def load_voc(self, vocfile) : 

		self.voc = Corpus.string(self, vocfile).split('\n')
		print("Vocabulary file loaded from current directory.")
		return self.voc

class Corpus(object) : 

	def __init__(self, path, voc="") : 
        
		self.corpus = path
		self.isfile = 0
		
		if choice == 'word' :
			if vocfile in glob("*") :
				self.voc = Voc.load_voc(self, vocfile)
				self.isfile = 1
		if choice == 'bigram' :	
			if bigramvocfile in glob("*") :
				self.voc = Voc.load_voc(self, bigramvocfile)
				self.isfile = 1
		if choice == 'trigram' :	
			if trigramvocfile in glob("*") :
				self.voc = Voc.load_voc(self, trigramvocfile)
				self.isfile = 1
				
		if self.isfile == 0  : 
			voc = Voc(self)
			voc.build_voc()
			self.voc = voc.voc
		self.bow = []
	

	def lire(self) :
		return glob(self.corpus+"/*")

	def string(self,vocfile) : 
		
		return open(vocfile).read()

	def getBOW(self) : 
		x = 0
		print("Creating BOW.")
		for fic in self.lire() :
			if x == nombredoc:
				break
			filebow = []
			wordlist = Voc.getfilewordlist(self, fic)
			if choice == 'bigram' :
				wordlist = Voc.getbigramlist(self, wordlist)
			if choice == 'trigram' :
				wordlist = Voc.gettrigramlist(self, wordlist)
			for item in self.voc :
				if item in wordlist :
					filebow.append('1')
				else :
					filebow.append('0')
			self.bow.append(filebow)
			x+=1
		if choice == 'word' :
			self.bowcsv(self.bow, "bow_"+vocfile[0:-3]+"tsv")
		if choice == 'bigram' :
			self.bowcsv(self.bow, "bow_"+bigramvocfile[0:-3]+"tsv")
		if choice == 'trigram' :
			self.bowcsv(self.bow, "bow_"+trigramvocfile[0:-3]+"tsv")

				
	def bowcsv(self, bow, filename) :
		with open(filename, 'w', newline='\n', encoding='utf-8') as file :
			writer = csv.writer(file, delimiter= '\t')
			writer.writerows(self.bow)
		print(f"Bag of words saved as \'{filename}\' in current directory.") 


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
	
	while True :
			choice = input("Choose between 'word', 'bigram' or 'trigram' BOW. Else, type 'exit'.\n")
			if choice == 'exit' :
				sys.exit()
			if choice == 'word' or choice == 'bigram' or choice == 'trigram' :
				break
			else : 
				print("This choice does not exist.")
				
	c = Corpus('../../corpus/imdb/*')
	c.getBOW()
