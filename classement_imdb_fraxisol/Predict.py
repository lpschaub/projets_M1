class Predict(object) : 

	def __init__(self,doc) : 
		self.doc = doc

	def predict(self) : 

		if 'good' in self.doc : 
			self.predicted = 'pos'
		else : 
			self.predicted = 'neg' 