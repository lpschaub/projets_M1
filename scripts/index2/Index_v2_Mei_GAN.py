
import glob
import spacy
import math
from collections import Counter

import pprint
nlp=spacy.load("fr_core_news_sm")

class Document(object) :

    def __init__(self,fic,lg='fr'):

        self.name = fic

        self.langue = lg

        self.content = open(fic,encoding='utf-8').read()#TODO close file
        
        self.infoLingui={}

    def tokenized(self) :

        self.tokens = self.content.split()

    def lemmatized(self) :
        sequence=nlp(self.content)
        for mot in sequence:        
            if mot.pos_ != 'PUNCT':
                if mot.text not in self.infoLingui.keys():
                    self.infoLingui[mot.text]={'lemma':mot.lemma_ }   
                else:
                    self.infoLingui[mot.text].update({'lemma':mot.lemma_ })
    

    def postaggued(self) :
        pass
                

    def lemma_pos(self) :
        string=nlp(self.content)
        for mot in string:  
            if mot.pos_ != 'PUNCT':        
                if mot.text not in self.infoLingui.keys():
                    self.infoLingui[mot.text]={'pos':mot.pos_}
                else:
                    self.infoLingui[mot.text].update({'pos':mot.pos_})
    
    def stopwordisation(self):
        string=nlp(self.content)
        for mot in string:  
            if mot.text not in self.infoLingui:
                if mot.is_stop==False and mot.pos_ != 'PUNCT':    
                    self.infoLingui[mot.text]={'isStopword':mot.is_stop}
            else:
                if mot.is_stop==True:
                    self.infoLingui.pop(mot.text)
                elif mot.pos_ != 'PUNCT':
                    self.infoLingui[mot.text].update({'isStopword':mot.is_stop})
    
    def lemma_token_stopword(self):
        
        string=nlp(self.content)
        list_words=[]
        for word in string:
            if word.pos_!= 'PUNCT' and word.is_stop==False:
                list_words.append(word.lemma_)
                
        return list_words
                
        
            
        
        

class Fable(Document) :

    def __init__(self,fic,lg='fr'):

        Document.__init__(self,fic,lg)

        self.titre = self.content.split('\n')[0]


# class Corpus(object) :


class Index(object) :

    def __init__(self,docs):

        self.dic = {}
        self.empty = True
        if type(docs) != list or len(docs) <2 :
            raise ValueError("Ce n'est pas une liste ou bien elle ne contient qu'un doc")

        for doc in docs :
            if not hasattr(doc,'titre') :
               doc.titre = doc.name
               
            doc.tokenized()
        self.docs = docs


    def maxFreq(self,mot):
        self.mot=mot
        
        if self.mot in self.dic.keys():
            nb=0
            for key,value in self.dic[mot].items():
                if value>nb:
                    nb=value
                    doc_name=key
                    
            return (doc_name, nb)
        else:
            raise ValueError("Il n'existe pas ce mot dans le corpus.") 


    def initialize(self):

        if self.empty :
            self.dic_list_words = {}
            for doc in self.docs :
                doc.lemmatized()
                doc.lemma_pos()
                doc.stopwordisation()
                self.dic_list_words[doc.name] = doc.lemma_token_stopword()
                Index.indexing(self,doc,doc.tokens)
            self.empty = False
            # pprint.pprint(self.dic)

        else :
            raise AttributeError("L'index n'est pas vide, vous ne pouvez pas l'initialiser")

    def blockIndex(self) :
        
        if not self.blocked:
            self.dic = {elem: [(cle, v[cle]) for cle in v] for (elem, v) in self.dic.items()}
            self.blocked = True

    def unBlockIndex(self):
        
        if self.blocked:
            self.dic = {elem:{tup[0]:tup[1] for tup in v} for (elem, v) in self.dic.items()}
            self.blocked = False


    def indexing(self,doc,tokens):
        if not tokens :
            self.words = [cle for cle in self.dic]
            return  0
        else :
            token = tokens[0]
            if token not in self.dic :
                self.dic[token] = {doc.titre: 1}
            else:
                if doc.titre not in self.dic[token]:
                    self.dic[token].update({doc.titre: 1})
                else:
                    self.dic[token][doc.titre] += 1
            try :
                return Index.indexing(self,doc,tokens[1:])
            except RecursionError :
                self.words = [cle for cle in self.dic]
                return 0
        
    def tf_idf(self, word_nb,word_sum, sum_file, word_file_sum):
        
        tf=(word_nb/word_sum)
        idf=math.log(sum_file/word_file_sum)
    
        return tf*idf
    
    def Tf_idf_in_words(self):
        
        #calculer le nombre de chque mot, le total des mots dans chque fichier 
        #calculer 
        sum_file=len(self.docs)
        
        self.dict_tf_Idf={}
        
        total_mots =[]
        
        for value in self.dic_list_words.values():
            total_mots.extend(list(set(value)))
        total_mot_counter=Counter(total_mots)
        # print(total_mot_counter)
        for key, value in self.dic_list_words.items():
            counter = Counter(value)
            word_sum=len(value)
            for mot, nombre in counter.items():
                if mot not in self.dict_tf_Idf.keys():
                    self.dict_tf_Idf[mot]={key:[nombre, self.tf_idf(nombre,word_sum,sum_file,total_mot_counter[mot])]}
                else:
                    self.dict_tf_Idf[mot].update({key:[nombre,self.tf_idf(nombre,word_sum,sum_file,total_mot_counter[mot])]})
            

    
    def donner_mot_tfIdf(self,word):
        self.word=word
        if self.word in self.dict_tf_Idf:
            print(self.dict_tf_Idf[self.word])
        else:
            print ("Le mot que vous avez saise n'exist pas dans le corpus.")
            





if __name__ == '__main__':
    
    #une liste des classe Fable
    fables = [Fable(doc) for doc in glob.glob('test2/*')]
    # dic_list_words = {}
    # for fable in fables:
    #     fable.lemmatized()
    #     fable.lemma_pos()
    #     fable.stopwordisation()
    #     dic_list_words[fable.name] = fable.lemma_token_stopword()
        


        
    
    
    index = Index(fables)
    
    index.initialize()
    index.Tf_idf_in_words()
    
    word_max = index.maxFreq('bon')
    print(word_max)
    print()
    
    print(index.dic_list_words)
    print()
    
    print(index.dict_tf_Idf)
    
    print()
    donner_mot=index.donner_mot_tfIdf('oui')
    
    
    
    
    
    #new_doc = Fable(glob.glob('new_doc/*.txt')[0])
    #new_doc.tokenized()
    
    #ajouter un nouveau fichier
    #index.indexing(new_doc,new_doc.tokens)
    
    #index.blockIndex()
    #index.unBlockIndex()
    
    # pprint.pprint(index.dic)
    # max  = index.maxFreq('courage')
    # index.initialize()



