import re
import nltk
from glob import glob
from nltk import ngrams


def listeMot(words):
	return dict([(word, True) for word in words])
liste = []
for fic in glob('../corpus/imdb/train/*/*.txt') : 
	if 'pos' in fic : 
		liste.append(((open(fic).read().split()), "pos"))
	else : 
		liste.append(((open(fic).read().split()), "neg"))

training_set = [(listeMot(txt), label) for (txt, label) in liste]
classif = nltk.NaiveBayesClassifier.train(training_set)
        
class Predict(object) : 


	def __init__(self, doc, seuil = 0.5) : 

		self.doc = doc
		self.seuil = seuil


	def predict(self) : 

		if classif.classify(listeMot((self.doc).split())) == "neg" :
			self.predicted = 'neg'
		elif classif.classify(listeMot((self.doc).split())) == "pos" : 
			self.predicted = 'pos'


if __name__ == '__main__':


	pred = Predict(open('../corpus/imdb/test/pos/8800_10.txt').read())
	pred.predict()
	print (pred.predicted)