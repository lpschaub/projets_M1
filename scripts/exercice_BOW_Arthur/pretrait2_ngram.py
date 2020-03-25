from glob import glob
import os
# from scripts.Evaluation import Evaluation
from PredictBOW import *
from Evaluation import Evaluation
import spacy
predicted = []
expected = []


nlp=spacy.load('en_core_web_sm')
# from Predict import Predict


import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score, classification_report

def pipeline(file) : 

    Corpus = pd.read_csv(file, sep='\t')
    # Step - a : Remove blank rows if any.
    

    Corpus['text'].dropna(inplace=True)# Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
    

    Corpus['text'] = [entry.lower() for entry in Corpus['text']]# Step - c : Tokenization : In this each entry in the corpus will be broken into set of words
    # Corpus['text']= [str(word_tokenize(entry)) for entry in Corpus['text']]# Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
    # tag_map = defaultdict(lambda : wn.NOUN)
    # tag_map['J'] = wn.ADJ
    # tag_map['V'] = wn.VERB
    # tag_map['R'] = wn.ADV
    # for index,entry in enumerate(Corpus['text']):
    #     # Declaring Empty List to store the words that follow the rules for this step
    #     Final_words = []
    # # Initializing WordNetLemmatizer()
    #     word_Lemmatized = WordNetLemmatizer()
    #     # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    #     for word, tag in pos_tag(entry):
    #         # Below condition is to check for Stop words and consider only alphabets
    #         if word not in stopwords.words('english') and word.isalpha():
    #             word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
    #             Final_words.append(str(word_Final))
    # # The final processed set of words for each iteration will be stored in 'text_final'
    # Corpus.loc[index,'text_final'] = str(Final_words)

    Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['text'],Corpus['label'],test_size=0.2)
    Encoder = LabelEncoder()
    # print(Train_Y)
    Train_Y = Encoder.fit_transform(Train_Y)
    # print(Train_Y)

    Test_Y = Encoder.fit_transform(Test_Y)
    Tfidf_vect = TfidfVectorizer(max_features=5000)
    Tfidf_vect.fit(Corpus['text'])
    Train_X_Tfidf = Tfidf_vect.transform(Train_X)
    Test_X_Tfidf = Tfidf_vect.transform(Test_X)
    # print(Tfidf_vect.vocabulary_)

    # fit the training dataset on the NB classifier
    Naive = naive_bayes.MultinomialNB()
    Naive.fit(Train_X_Tfidf,Train_Y)# predict the labels on validation dataset
    predictions_NB = Naive.predict(Test_X_Tfidf)# Use accuracy_score function to get the accuracy
    # print(predictions_NB)
    print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)

    print(classification_report(Test_Y,predictions_NB, labels=[0,1], target_names=['neg','pos']))
    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    SVM.fit(Train_X_Tfidf,Train_Y)# predict the labels on validation dataset
    predictions_SVM = SVM.predict(Test_X_Tfidf)# Use accuracy_score function to get the accuracy
    print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)
    print(classification_report(Test_Y,predictions_SVM, labels=[0,1], target_names=['neg','pos']))



def file2csv(text, label):
	
	with open("en-tete.csv", 'w', encoding='utf-8') as file :
		for item, item2 in zip(text, label) :
			file.write(f"{item}\t{item2}\n")



def getFile(fic,pol): 
    x = 0
    for f in glob(fic+'/*') : 
        x += 1
        if x >= 2000 : 
            os.system(f'mv {f} /home/schaub/Téléchargements/aclImdb_v1/aclImdb/test/{pol}/')

        if x == 8000 : 
            break


def getcontentlabel(file) :
    
    expect = '' 
    if '/pos/' in file : 
        expect = 'pos'
    else : 
        expect = 'neg'

    return open(file).read(),expect


def prediction(file,expect) : 


    pred = Predict(file, pw, nw, nlp)
    pred.predict()
    predicted.append(pred.predicted)
    expected.append(expect)

def clean(ficstring) : 
	exception = [".", ";", ":", ",", "!", "?", "(", ")", ">", "<", "\"", "...", "/", "-", "_", "\t", ""]
	ficstring = re.sub("<br|/>", '', ficstring)
	for item in exception :
		ficstring = ficstring.replace(item,'')
	ficstring = ficstring.replace('\n',' ')
	return ficstring

if __name__ == '__main__':

		text = []
		label = []
		for file in glob('../../corpus/imdb/**/*.txt') :
			content = getcontentlabel(file)
			text.append(clean(content[0]))
			label.append(content[1])
		file2csv(text, label)
    