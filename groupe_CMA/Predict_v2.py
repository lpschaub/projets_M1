#auteur : CMA
#version 2

import nltk
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer() 


class Predict(object) : 
    
    def __init__(self,doc) : 
        self.doc = doc
        self.pos_emotion=['classic','enjoy','good','impress','pleas','interest','funni','thrill',"excel",
                          'comfort','amaz','surprise','sure','brilliant','unusu','special','wel','must','wonder','best','superb']
        self.neg_emotion=["don't","doesn't","infamous","unfortun","too",'ever','impossible',"bore","3-D",'hard',
                          "aw","strange",'disaster','uncomfort','never','stupid',"can't","terri",'hollow','message-less',"n't","either"]
        self.super_pos_emotion=['love','great','beauti','howev','fantast','perfect','favorit',"top",]
        self.super_neg_emotion=['could','should','littl','noth','worst','bad','poor',"fail"]
        
        
    def countScore(self,score=0.0):
        self.score=score 
        sens=nltk.sent_tokenize(self.doc)
        for sent in sens:
            words=[]
            words.extend(nltk.word_tokenize(sent.lower()))
        
            self.words_list=[]
            for word in words:
                if word not in ['!',',','.','?','>','<','/',':','(',')','<','>','...','``',"'"]:
                    stemmer=porter_stemmer.stem(word)
                    self.words_list.append(stemmer)
					
            for word in self.words_list:
                if word in self.super_pos_emotion:
                    self.score += 0.2
                if word in self.super_neg_emotion:
                    self.score -= 0.2
                if word in self.pos_emotion:
                    self.score += 0.1
                if word in self.neg_emotion:
                    self.score -= 0.1
            
        
    def predict(self):
        self.countScore()
        if self.score >=0.0:
            self.predicted='pos'
        else:
            self.predicted='neg'
            
                
if __name__ == '__main__':

	pred = Predict(open('../corpus/imdb/pos/10859_7.txt').read())
	pred.predict()
	
	print (pred.predicted)
