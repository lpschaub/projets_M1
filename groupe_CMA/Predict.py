import re

class Predict(object) : 


	def __init__(self, doc, seuil = 0.5) : 

		self.doc = doc
		self.seuil = seuil


	def workspace(self) : 

		if 'good' in self.doc : 
			self.seuil += 0.1
		else : 
			self.seuil  -= 0.1

		if 'excellent' in self.doc : 
			self.seuil += 0.1
		else : 
			self.seuil  -= 0.1

		if 'bad' in self.doc and not 'not bad' in self.doc : 
			self.seuil -= 0.1
		else : 
			self.seuil  += 0.1

		if 'convincing' in self.doc : 
			self.seuil += 0.1
		else : 
			self.seuil -= 0.1

		pattern1 = re.compile(r'!{2}') # au moins deux points d'exclamation qui se suivent 
		pattern2 = re.compile(r'[A-Z]{2}[A-Z]+') # un mot en majuscule

		res = re.search(pattern1, self.doc)
		res2 = re.search(pattern2, self.doc)

		if res and res2 : # la combinaison des deux provoque une baisse exponentielle du seuil (0.3 au lieu de 0.2)

			self.seuil -= 0.3 

		elif res : #s'il y'a seulement les exclamations

			self.seuil -= 0.1 

		elif res2 : #s'il y'a seulement les mots en majuscule

			self.seuil -= 0.1

		else : 
			self.seuil = self.seuil #sinon le seuil reste inchangé

		


	def predict(self) :

		self.workspace()

		if self.seuil < 0.5 : 
			self.predicted = 'neg'
		else : 
			self.predicted = 'pos'


if __name__ == '__main__':


	pred = Predict(open('../corpus/imdb/pos/33_7.txt').read())
	pred.predict()
	print (pred.predicted)