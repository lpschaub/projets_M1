import re
import spacy
nlp=spacy.load('en_core_web_sm')
class Predict(object) :


	def __init__(self, doc, seuil = 0.5) :

		self.doc = doc
		self.seuil = seuil
		self.nlpdoc = nlp(self.doc)
		self.lemmas = [token.lemma_ for token in self.nlpdoc]
		self.lemmatized=" ".join(self.lemmas)

	def workspace(self) :

		mots_pos=["beautiful","likable","engaging","impressive","sensual","genuine","fresh","mysterious","authentic","interesting","fun","nice","favourite","favorite","funny","colourful","charming","charmingly","inventive","innovative","moving","entertaining","intelligent","amusing","nicely","good","solid","memorable","impressed"]
		mots_pos_plus=["fascinating","unforgettable","hilarious","fantastic","thrilling","recommended","amazing","excellent","excellently","delightful","marvellous","brilliant","delight","finest","joyous","genius","masterfully","great","greatest","masterpiece","awesome","sublime","wonderful","excellently","witty","irresistible","best","must-see","must see","praise"]
		for mot in mots_pos:
			match=re.findall(rf"\w+ \w+ \w+ {mot} ", self.lemmatized)
			for m in match:
				if "not" in m or "no " in m or "neither " in m:
					self.seuil-=0.1
					#print(m+"-1")
				else:
					self.seuil+=0.1
					#print(m+"+1")
		for mot in mots_pos_plus:
			match=re.findall(rf"\w+ \w+ \w+ {mot} ", self.lemmatized)
			for m in match:
				if "not" in m or "no " in m or "neither " in m:
					self.seuil-=0.2
					#print(m+"-2")
				else:
					self.seuil+=0.2
					#print(m+"+2")




	def predict(self) :

		self.workspace()
		#print(self.seuil)

		if self.seuil < 0.6 :
			self.predicted = 'neg'
		else :
			self.predicted = 'pos'


if __name__ == '__main__':


	pred = Predict(open('../corpus/imdb/pos/33_7.txt').read())
	pred.predict()
	print (pred.predicted)
