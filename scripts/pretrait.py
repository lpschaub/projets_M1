from glob import glob
import os
# from scripts.Evaluation import Evaluation
from Predict import *
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
    corpus = Corpus.reindex(np.random.permutation(Corpus.index))
    print(corpus.head())
    # Step - a : Remove blank rows if any.
    print(corpus['text'])
    print(corpus['label'])

    Corpus['text'].dropna(inplace=True)# Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
    
    print(type(Corpus['text']))
    for elem in Corpus['text'] : 
        print(elem)
        print(type(elem))
        break

    Corpus['text'] = [entry.lower() for entry in Corpus['text']]# Step - c : Tokenization : In this each entry in the corpus will be broken into set of words
    

    Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['text'],Corpus['label'],test_size=0.2)
    print(len(Train_X))
    print(len(Train_Y))

    Encoder = LabelEncoder()
    Train_Y = Encoder.fit_transform(Train_Y)
    print(Train_Y)

    Test_Y = Encoder.fit_transform(Test_Y)
    print(Test_Y)
    sys.exit()


    representation = CorpusCSV(Corpus['text'])
    voc = representation.voc
    Train_X = CorpusCSV(Train_X, voc)
    Test_X = CorpusCSV(Test_X, voc)
    Train_X.getBOW()
    Test_X.getBOW()
    Train_X_Tfidf = Train_X.bow
    Test_X_Tfidf = Test_X.bow
    print(Train_Y)
    # Tfidf_vect = TfidfVectorizer(max_features=5000)
    # Tfidf_vect.fit(Corpus['text'])
    # Train_X_Tfidf = 
    # Test_X_Tfidf = Tfidf_vect.transform(Test_X)
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





def files2csv(text,out,label):
    
    
    text = text.replace('\t',"")
    text = text.replace('\n',"")
    print(label)
    out.write(text+'\t'+label+'\n')




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
    if 'p/' in file : 
        expect = 'pos'
    else : 
        expect = 'neg'

    return open(file).read(),expect


def prediction(file,expect) : 


    pred = Predict(file, pw, nw, nlp)
    pred.predict()
    predicted.append(pred.predicted)
    expected.append(expect)


if __name__ == '__main__':

    out = open('../data/dataset1.csv',encoding='utf-8')
    # out.write('text\tlabel\n')
    pipeline(out)
    # x = 0
    # y = 0
    # for fic in glob('/home/schaub/Téléchargements/aclImdb_v1/aclImdb/test/p/*.txt') : 
    #     print(fic)
    #     x += 1
    #     file, label = getcontentlabel(fic)
    #     files2csv(file,out,label)
    #     if x == 100 : 
    #         break
    # for fic in glob('/home/schaub/Téléchargements/aclImdb_v1/aclImdb/test/n/*.txt') :
    #     print(fic)
    #     y += 1
    #     file, label = getcontentlabel(fic)
    #     files2csv(file,out,label)
    #     if y == 1000 : 
    #         break


        # print(file)
        # print(label)

        # prediction(file,label)


    # print(len(predicted))
    # print(len(expected))

    # eval = Evaluation(expected,predicted)

    
    # print(eval.getVraisPos())
    # print(eval.getFauxNeg())
    # print(eval.getFauxPos())
    # print(eval.f_mesure())