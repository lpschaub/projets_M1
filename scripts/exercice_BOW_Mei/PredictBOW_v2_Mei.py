from glob import glob
import re
import spacy
from spacy import displacy
import pandas as pd
from collections import Counter
import math
# nlp = spacy.load('en')

class Voc(object) : 

    def __init__(self, corpus) :

        self.corpus = corpus
        self.voc = []
        self.biGram = []
        self.triGram = []

    def build_voc(self) : 
        for fic in self.corpus.lire() : 
            ficstring = open(fic).read()
            ficstring = self.clean(self.corpus.string(fic).replace('\n',' '))
            fics = ficstring.split()
            self.voc.extend(fics)
        
        return list(set(self.voc))
	
    def clean(self,ficstring) : 
        ficstring = ficstring.replace('(','')
        ficstring = ficstring.replace(')','')
        ficstring = ficstring.replace('<','')
        ficstring = ficstring.replace('>','')
        
        
        return ficstring


    def ecrire_voc(self,out) : 

        out.write('\n'.join([elem for elem in self.voc]))

    def load_voc(self, vocfile) : 

       # self.voc = Voc.string(vocfile)
       self.voc = Corpus.string(vocfile)
       
#csv生成vocabulaire
class VocCSV(Voc) :

    def __init__(self, corpus) :
        Voc.__init__(self, corpus)

        
    def build_voc(self): 
        read_file = pd.read_csv(self.corpus, sep='\t')
        lis = read_file['text']
        for line in lis:
            word_lis = line.split()
            self.voc.extend(word_lis)
        # for lines in self.corpus : 
        #     for elem in lines.split() : 
        #         self.voc.append(elem)
        self.voc = list(set(self.voc))
		# print(self.voc)
    
    def build_bi_gram(self):
        read_file = pd.read_csv(self.corpus, sep='\t')
        lis = read_file['text']
        for line in lis:
            word_list = line.split()
            for i in range (len(word_list)-1):
                # print(type(word_list[i]))
                bigram = word_list[i]+' '+word_list[i+1]
                # print(bi_text)
                self.biGram.append(bigram)
                
    def build_tri_gram(self):
        read_file = pd.read_csv(self.corpus, sep='\t')
        lis = read_file['text']
        for line in lis:
            word_list = line.split()
            for i in range (len(word_list)-2):
                # print(type(word_list[i]))
                trigram = word_list[i]+' '+word_list[i+1]+' '+word_list[i+2]
                # print(bi_text)
                self.triGram.append(trigram)
            
        
        



class Corpus(object) : 

    def __init__(self, path, voc = "") : 

        self.corpus = path
        if voc : 
            self.voc = Voc.load_voc(voc)
        else : 
            voc = Voc(self)
            voc.build_voc()
            self.voc = voc.voc
        self.bow = []

    def lire (self) :
        return glob(self.corpus+"/*")


    def string(self,fic) : 

        return open(fic).read()
    '''
    def clean(self,ficstring) : 
        ficstring = ficstring.replace('(','')
        ficstring = ficstring.replace(')','')
        ficstring = ficstring.replace('<','')
        ficstring = ficstring.replace('>','')
        
        return ficstring
    '''
    def getBOW(self,inp) : 
        # x = 0
        self.inp = inp
        voc = Voc(self)
        for fic in glob(self.inp) : 
            list_words = []
            fics = open(fic).read()
            words = voc.clean(fics.replace('\n',' '))
            contenu = words.split()
            for word in contenu:
                if word in self.voc:
                    list_words.append(1)
                else:
                    list_words.append(0)
            
            
            self.bow.append(list_words)

class CorpusCSV(Corpus) : 
    def __init__(self, path, voc = "") : 
        self.corpus = path
        if voc : 
            self.voc = voc
        else : 
            voc = VocCSV(path)
            voc.build_voc()
            voc.build_bi_gram()
            voc.build_tri_gram()
            self.voc = voc.voc
            self.biGram = voc.biGram
            self.triGram = voc.triGram
            
            # print(len(self.voc))
        self.bow = []
        
    def string(self, fic) : 
        return self.corpus

    def lire(self): 
        read_file = pd.read_csv(self.corpus, sep='\t')
        self.corpus = read_file['text']
        return self.corpus
    
    def getBow(self):
        self.dire()
        bow = []
        nb = 1
        # print(self.voc)
        for line in self.corpus:
            word_list = line.split()
            for word in self.voc:
                if word in word_list:
                    bow.append(1)
                else:
                    bow.append(0)
            print (f"doc{nb}\t{bow}\n")
            # break
            # print(len(word_list))
            print("*****")
            bow=[]
            nb+=1
            
    def search_bi_gram(self):
        self.lire()
        bi = []
        nb = 1
        for line in self.corpus:
            word_list = line.split()
            bigram_line = []
            for i in range (len(word_list)-1):
                bigram = word_list[i]+' '+word_list[i+1]
                bigram_line.append(bigram)
            for word in self.biGram:
                if word in bigram_line:
                    bi.append(1)
                else:
                    bi.append(0)
            print (f"doc{nb}\t{bi}\n")
            # break
            # print(len(word_list))
            print("*****")
            bi=[]
            nb+=1
            
    #search_tri_gram
    def search_tri_gram(self):
        self.lire()
        tri = []
        nb = 1
        for line in self.corpus:
            # print(line)
            word_list = line.split()
            trigram_line = []
            for i in range (len(word_list)-2):
                trigram = word_list[i]+' '+word_list[i+1]+' '+word_list[i+2]
                trigram_line.append(trigram)
            for word in self.triGram:
                if word in trigram_line:
                    # print(word)
                    tri.append(1)
                else:
                    tri.append(0)
            print (f"doc{nb}\t{tri}\n")
            # print(len(tri))
            # break
            # print(len(word_list))
            print("*****")
            tri=[]
            nb+=1
            
    def tfIdf(self, occurrence, sumNb,sumFile,occurrenceFile):
        tf = occurrence/sumNb
        idF = math.log(sumFile/occurrenceFile)
        return tf*idF
    
#tfidf pour un mot
    def tfIdf_bow(self):
        count=1
        self.lire()
        #先拿到每一个单词出现在多少个文档
        wordInFile =[]
        for line in self.corpus:
            word_list = line.split() 
            wordInFile.extend(set(word_list))
        wordInFile_dic = Counter(wordInFile)
        sumFile = len(self.corpus)-1
        
        for line in self.corpus:
            idIdf_line=[]
            word_list = line.split()
            word_dic = Counter(word_list)
            sumNb = len(word_list)
            for item in self.voc:
                if item in word_list:
                    tfIdf_Mot = (f"{self.tfIdf(word_dic[item],sumNb,sumFile, wordInFile_dic[item]):.4f}")
                    idIdf_line.append(tfIdf_Mot)
                else:
                    idIdf_line.append(0)
            print(f"doc{count} : {idIdf_line}\n")
            print("*****")
            count+=1
        
#tfidf pour un bi-gram
    def tfIdf_biText(self):
        count=1
        self.lire()
        #先拿到每一个bitext出现在多少个文档
        wordInFile_list = []
        for line in self.corpus:
            word_list = line.split()
            bigram_line = []
            for i in range (len(word_list)-1):
                bigram = word_list[i]+' '+word_list[i+1]
                bigram_line.append(bigram)
            wordInFile_list.extend(set(bigram_line))

        wordInFile_dic = Counter(wordInFile_list)
        sumFile = len(self.corpus)-1
        
        for line in self.corpus:
            word_list = []
            idIdf_line = []
            word_list = line.split()
            bigram_line = []
            for i in range (len(word_list)-1):
                bigram = word_list[i]+' '+word_list[i+1]
                bigram_line.append(bigram)
            word_dic = Counter(bigram_line)
            sumNb = len(bigram_line)
            for item in self.biGram:
                if item in bigram_line:
                    # print(item)
                    tfIdf_Mot = (f"{self.tfIdf(word_dic[item],sumNb,sumFile, wordInFile_dic[item]):.4f}")
                    # print(tfIdf_Mot)
                    idIdf_line.append(tfIdf_Mot)
                else:
                    idIdf_line.append(0)
            print(f"doc{count} : {idIdf_line}\n")
            print("*****")
            count+=1
            # break
        
        
#tfidf pour un tri-gram
    def tfIdf_triText(self):
        count=1
        self.lire()
        #先拿到每一个tritext出现在多少个文档
        wordInFile_list = []
        for line in self.corpus:
            word_list = line.split()
            trigram_line = []
            for i in range (len(word_list)-2):
                trigram = word_list[i]+' '+word_list[i+1]+' '+word_list[i+2]
                trigram_line.append(trigram)
            wordInFile_list.extend(set(trigram_line))

        wordInFile_dic = Counter(wordInFile_list)
        sumFile = len(self.corpus)-1
        
        for line in self.corpus:
            word_list = []
            idIdf_line = []
            word_list = line.split()
            trigram_line = []
            for i in range (len(word_list)-2):
                trigram = word_list[i]+' '+word_list[i+1]+' '+word_list[i+2]
                trigram_line.append(trigram)
            word_dic = Counter(trigram_line)
            sumNb = len(trigram_line)
            for item in self.triGram:
                if item in trigram_line:
                    # print(item)
                    tfIdf_Mot = (f"{self.tfIdf(word_dic[item],sumNb,sumFile, wordInFile_dic[item]):.4f}")
                    # print(tfIdf_Mot)
                    idIdf_line.append(tfIdf_Mot)
                else:
                    idIdf_line.append(0)
            print(f"doc{count} : {idIdf_line}\n")
            print("*****")
            count+=1
            # break

# class Predict(object) : 


#  	def __init__(self, doc, nlp, seuil = 0.5) : 

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
    
    # c = Corpus('../corpus/imdb/neg')
    # c.getBOW('../corpus/imdb/pos/10628_7.txt')
    #print(c.bow)
    
    #读取csv文件并生成voc
    voc_csv=CorpusCSV('out.csv')
    # print(voc_csv.voc)
    # voc_csv.getBow()
    # voc_csv.search_bi_gram()
    #读取每一行的然后对照是不是在voc里面
    
    #itidfisation
    # voc_csv.tfIdf_bow()
    # voc_csv.tfIdf_biText()
    voc_csv.tfIdf_triText()
    
