# KRISTINA SOPHIE CAMILLE

import re
#import spacy
#nlp=spacy.load('en_core_web_sm')
class Predict(object) :


	def __init__(self, doc, nlp, seuil = 0.5) :

		self.doc = doc
		self.seuil = seuil
		self.nlpdoc = nlp(self.doc)
		self.lemmas = [token.lemma_ for token in self.nlpdoc]
		self.lemmatized=" ".join(self.lemmas)

	def workspace(self) :

		mots_pos=["beautiful","likable","engaging","impressive","sensual","genuine","fresh","mysterious","authentic","interesting","fun","nice","favourite","favorite","funny","colourful","charming","charmingly","inventive","innovative","moving","entertaining","intelligent","amusing","nicely","good","solid","memorable","impressed","wow","enjoyable","enjoy"]
		mots_pos_plus=["should see","should watch","recommend","fascinating","unforgettable","hilarious","fantastic","thrilling","recommended","amazing","excellent","excellently","delightful","marvellous","brilliant","delight","finest","joyous","genius","masterfully","great","greatest","masterpiece","awesome","sublime","wonderful","excellently","witty","irresistible","best","must-see","must see","praise","outstanding","breathtaking","terrific","convincing","convinced"]
		mots_neg=["annoying","off putting","ugly","weird","unfortunately","waste \w+ time","ridiculous","worse","stupid","predictable","silliest","disappoint","laughable","can not stand","screw","sketchy","non-existent","more or less","in \w+ opinion","no idea","too","attempt","long"]
		mots_neg_plus=["awful","crappy","terrible","badly","bad","poorly","shit","pathetic","disappointing","disappointed","hate","dislike","failure","mediocre","mediocrity","worst","fail","cliche","cliché","painful","painfully","should not watch","dull","cornball","avoid watch","irritating","overrated","boring","lame","miserably","zero effort","brainless","problem","issue","make no sense","inaccurate","cringy","shallow","cheesy","repetitive","tedious"]
		expression_neg=["could have been","should have been","nothing more than","over and over","the problem is that","waste of time","the whole thing","to begin with","might as well","what the hell","the whole movie","too bad that","saving grace","not to say","of the worst"]
		intensifiers=re.compile(r'really|very|incredibly|so|highly|extremely|just|utterly')
		for mot in mots_pos:
			match=re.findall(rf"\w+ \w+ \w+ {mot} ", self.lemmatized)
			for m in match:
				if "not" in m or "no " in m or "neither " in m or "only" in m or "would be" in m or "could have been" in m or "would have" in m:
					self.seuil-=0.2
					#print(m+"-1")
				else:
					if re.search(intensifiers, m):
						self.seuil+=0.3
					else:
						self.seuil+=0.15
					#print(m+"+1")
		for mot in mots_pos_plus:
			match=re.findall(rf"\w+ \w+ \w+ {mot} ", self.lemmatized)
			for m in match:
				if "not" in m or "no " in m or "neither " in m or "only" in m or "would be" in m or "could have been" in m or "would have" in m:
					self.seuil-=0.3
					#print(m+"-2")
				else:
					if re.search(intensifiers, m):
						self.seuil+=0.4
					else:
						self.seuil+=0.2
					#print(m+"+2")

		for mot in mots_neg:
			match=re.findall(rf"\w+ \w+ \w+ {mot} ", self.lemmatized)
			for m in match:
				if "not" in m or "no " in m or "neither " in m:
					self.seuil+=0.05
					#print(m+"-1")
				else:
					if re.search(intensifiers, m):
						self.seuil-=0.4
					else:
						self.seuil-=0.2
					#print(m+"+1")
		for mot in mots_neg_plus:
			match=re.findall(rf"\w+ \w+ \w+ {mot} ", self.lemmatized)
			for m in match:
				if "not" in m or "no " in m or "neither " in m:
					self.seuil+=0.05
					#print(m+"-2")
				else:
					if re.search(intensifiers, m):
						self.seuil-=0.6
					else:
						self.seuil-=0.3
					#print(m+"+2")
		for mot in expression_neg:
			match=re.findall(rf"{mot}", self.doc)
			for m in match:
				self.seuil-=0.3
					#print(m+"+2")
		bonne_note=re.compile(r'[6-9]/10')
		mauvaise_note=re.compile(r'[0-5]/10')
		if re.search(bonne_note, self.doc):
			self.seuil+=0.2
			#print("note explicite positive dans : ")
			#print(self.doc)
		if re.search(mauvaise_note, self.doc):
			self.seuil-=0.4
			#print("note explicite negative dans : ")
			#print(self.doc)


	def predict(self) :

		self.workspace()
		#print(self.seuil)

		if self.seuil < 0.5 :
			self.predicted = 'neg'
		else :
			self.predicted = 'pos'


if __name__ == '__main__':


	pred = Predict(open('../corpus/imdb/pos/33_7.txt').read())
	pred.predict()
	print (pred.predicted)
