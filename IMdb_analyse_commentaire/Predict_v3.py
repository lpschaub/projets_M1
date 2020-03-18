#auteur : CMA
#version 3
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
porter_stemmer = PorterStemmer() 

class Predict(object) : 
    
    def __init__(self,doc) : 
        self.doc = doc
        
        
    def countScore(self,score=0,pos_avey=0,neg_avey=0):
        sid = SentimentIntensityAnalyzer()
        self.score=score 
        sens=nltk.sent_tokenize(self.doc)
        for sent in sens:
            
            # print(sent)
            ss=sid.polarity_scores(sent)
            
            neg=ss['neg']
            pos=ss['pos']
            if neg-pos > 0 and neg-pos < 0.1:
                self.score-=0.06
            elif neg-pos >= 0.1 and neg-pos < 0.2:
                self.score-=0.11
            elif neg-pos >= 0.2 and neg-pos < 0.3:
                self.score-=0.15
            elif neg-pos >= 0.3 and neg-pos < 0.4:
                self.score-=0.2
            elif neg-pos >= 0.4:
                self.score-=0.22
            elif pos-neg > 0 and pos- neg < 0.1:
                self.score+=0.05
            elif pos-neg >= 0.1 and pos- neg < 0.2:
                self.score+=0.07
            elif pos-neg >= 0.2 and pos-neg < 0.3:
                self.score+=0.08
            elif pos-neg >= 0.3 and pos-neg < 0.4:
                self.score+=0.12
            elif pos-neg >= 0.4 and pos-neg < 0.5:
                self.score+=0.14
            elif pos-neg >= 0.5:
                self.score+=0.16
            elif neg == pos:
                continue
                
                
            
    def predict(self):
        self.countScore()
                      
        if self.score >0.0:
            self.predicted='pos'
        else:
            self.predicted='neg'
            
                
if __name__ == '__main__':

 	  pred = Predict(open('../corpus/imdb/neg/3204_3.txt').read())
 	  pred.predict()

 	  print (pred.predicted)
      
     

