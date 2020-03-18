#auteur : CMA
#version 1
import nltk


from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer() 


class Predict(object) : 
    
    def __init__(self,doc) : 
        self.doc = doc
        self.pos_emotion=['classic','great','enjoy','good','impress','pleas','interest','funni','thrill',"3-D","excel",
                          'comfort','amazingli','amaz','wonder','surprise','sure','best','brilliant','unusu','special']
        self.neg_emotion=['bad',"don't","doesn't","infamous","unfortun","bore","too",'ever','impossible',
                          "aw","strange",'disaster','uncomfort',"n't",'never','stupid',"can't","terri",'hollow','message-less']
        
        
    def traiteDonee(self):
        words=[]
        sens=nltk.sent_tokenize(self.doc)
        for sent in sens:
            words.extend(nltk.word_tokenize(sent))
        
        self.words_list=[]
        for word in words:
            if word not in ['!',',','.','?','>','<','/',':','(',')','<','>','...','``',"'"]:
                stemmer=porter_stemmer.stem(word)
                self.words_list.append(stemmer)
        
        
    
    def countScore(self,score=0):
        self.score=score 
        sens=nltk.sent_tokenize(self.doc)
        for sent in sens:
            words=[]
            words.extend(nltk.word_tokenize(sent))
        
            self.words_list=[]
            for word in words:
                if word not in ['!',',','.','?','>','<','/',':','(',')','<','>','...','``',"'"]:
                    stemmer=porter_stemmer.stem(word)
                    self.words_list.append(stemmer)
        
            for word in self.words_list:
                if word in self.pos_emotion:
                    print(word)
                    self.score += 0.1
                if word in self.neg_emotion:
                    print(word)
                    self.score -= 0.1
            print(self.score)
        
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