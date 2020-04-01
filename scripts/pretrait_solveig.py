from glob import glob
import os, sys, time
# from scripts.Evaluation import Evaluation
#from Predict import *
#from Evaluation import Evaluation
import spacy
predicted = []
expected = []
import collections
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


def normal(elem) : 
    return elem.split()

def tokens(elem) :
    return ' '.join([token.text for token in nlp(elem)if token.text != ''] ) #join permet de faire tous les traitements séparément et dans n'importe quel ordre

def lemmas(elem) : 
    return ' '.join([token.lemma_ for token in nlp(elem)if token.text != ''] )   

def swds(elem) :
    return ' '.join([token.text for token in nlp(elem) if not token.is_stop and token.text != '' ])

def swdslemma(elem) : 
    return ' '.join([token.lemma_ for token in nlp(elem) if not token.is_stop and token.text != '' ])

def clean(elem) : 

    elem = elem.replace('"','')
    elem = elem.replace('(','')
    elem = elem.replace(')','')
    elem = elem.replace('[','')
    elem = elem.replace(']','')
    elem = elem.replace('  ',' ')
    elem = elem.replace('\t',' ')
    elem = elem.replace('<','')
    elem = elem.replace('>','')
    elem = elem.replace("'s",' is')
    elem = elem.replace('-','')
    elem = elem.replace('/br','')
    elem = elem.replace('?','')
    elem = elem.replace(':','')
    elem = elem.replace('...','')
    elem = elem.replace('!','')
    elem = elem.replace('/','')
    elem = elem.replace('\\','')
    elem = elem.replace('>','')
    elem = elem.replace(';',' ')
    elem = elem.replace('/>','')
    

    return ' '.join([word for word in elem.split() if word != ''])

def maxmin_df(corpus, mindf = 0.05, maxdf = 0.95) : #à faire après les autres prétraitement et, surtout, après création de n-gram ! (sinon corpus est une liste de strings)

    """
    On max min dfise, ce sont des paramètres que vous pouvez changer dans l'appel de la fonction dans pipeline()
    """ 

    N = len(corpus) #c'est une liste de listes (si liste de strings, set renvoie un ensemble de caractères et non de mots)
    voc = {}
    final = []
    for elem in corpus : 
        elem = list(set(elem)) #parce qu'on doit compter le nb de doc où un mot apparaît (si plusieurs mots dans un doc, on va compter doc plusieurs fois ensuite donc on retire les doublons dans chaque doc pour éviter ça)
        for word in elem :
            if word not in voc : 
                voc[word] = 1
            else :
                voc[word] += 1
    for mot in voc : 
        if voc[mot]/N > mindf and voc[mot]/N < maxdf and mot != '' : 
            final.append(mot)

    new_corpus = []
    for elem in corpus : 
         new_corpus.append([token for token  in elem if token in final])
    return final, new_corpus #final : liste non ordonnée et sans doublons (voc), new_corpus liste de listes ordonnée et avec doublons


def voca(corpus) : #renvoie juste la taille du voc, le voc étant renvoyé par la fonction précédente

    voc = []

    for elem in corpus : 
        for token in elem : #token peut être un mot ou un n-gram selon le corpus que l'on passe en paramètre
            if token != '' : 
                voc.append(token)
            else : 
                print(token) #à quoi sert cette ligne ?
    return len(list(set(voc)))



def simple_bow(corpus, voc) :

    bow = []
    i = 0
    for elem in corpus : 
        bow.append([])
        for word in voc : 
            if word in elem : 
                bow[i].append(1)
            else : 
                bow[i].append(0)
        i += 1
    return bow

def frequency_bow(corpus, voc) : 
    bow = []
    i = 0

    for elem in corpus : 
        elem = collections.Counter(elem) #crée automatiquement un dico Counter qui donne le nb d'occurrences de chaque élément (mot) de la liste élém
        bow.append([])
        for word in voc : 
            if word in elem : 
                bow[i].append(elem[word])
            else : 
                bow[i].append(0)
        i += 1
    # sys.exit()
    return bow
    
def tfidf_bow(corpus, corpus_train, voc):
    bow = []
    i = 0

    for elem in corpus :
        L = len(elem)
        elem = collections.Counter(elem)
        bow.append([])
        for word in voc : 
            if word in elem :
                idf = getIDF(corpus_train, word)
                bow[i].append(elem[word]/L*idf)
            else : 
                bow[i].append(0)
        i += 1
    return bow

def getIDF(corpus_train, ngram):
    N = len(corpus_train)
    F = 0
    for elem in corpus_train : 
        if ngram in elem : 
            F += 1
    #print(F)
    return N/F
    

from nltk import ngrams


def n_grammeur(corpus, n_grammes) :
    """
    On transforme le corpus en n-grammes (on généralise et on passe le n-grammes en paramètres de la fonction)
    """
    new_corpus = []

    for elem in corpus : 
        new_elem = '' # à quoi sert cette ligne ?? (cette variable n'est pas du tout utilisée ensuite)
        elem = elem.split()
        new_corpus.append([grams for grams in ngrams(elem, n_grammes)])
    return new_corpus



# def frequency_bow(corpus, voc) :



def pipeline(file, n_grammes = 2) : 

    """
        Par défaut, on veut des bigrammes. 
        VOus choisissez ici quel prétraitement effectuer (lower case, lemmatisation stopwords.. etc vous êtes libres d'en rajouter)
    """

    Corpus = pd.read_csv(file, sep='\t')
    # has_voc = False



    print("########################################################\n\n")

    print('Pré-taitement du corpus \n\n')
    # Step - a : Remove blank rows if any.
    Corpus['text'].dropna(inplace=True)

    # Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently ##ligne 183??
    Corpus['text'] = [entry.lower() for entry in Corpus['text']]
    print('Taille du corpus avant nettoiement : ' +str(sum([len(elem.split()) for elem in Corpus['text']]))+'\n\n')
    
    # Step - c : Tokenization : In this each entry in the corpus will be broken into set of words
    # Corpus['text'] = [swds(elem) for elem in Corpus['text']]   # On stopwordise en tokens (si vous ne voulez pas, il faut commenter cette ligne)

    Corpus['text'] = [clean(elem) for elem in Corpus['text']]
    print('Taille du corpus avant prétraitement : ' +str(sum([len(elem.split()) for elem in Corpus['text']]))+'\n\n')
    # Corpus['text'] = [swds(elem) for elem in Corpus['text']]   # On stopwordise en tokens (si vous ne voulez pas, il faut commenter cette ligne)

    Corpus['text'] = [swdslemma(elem) for elem in Corpus['text']]   # On stopwordise en lemmes (si vous ne voulez pas, il faut commenter cette ligne)

    #Corpus['text'] = [lemmas(elem) for elem in Corpus['text']]   # On lemmatise sans stopwordiser (si vous ne voulez pas, il faut commenter cette ligne)

    print("########################################################\n\n")
    new = []
    # On nettoie ici le corpus une dernière fois pour virer les char qui se balladent et qui faussent les résultats
    for elem in Corpus['text'] : 
        new_el= []
        for word in elem.split(): 
            if len(word) > 1 : 
                new_el.append(word)
        new.append(' '.join(new_el))
    Corpus['text'] = new

    print('Taille du corpus après  prétraitement : '+str(sum([len(elem.split()) for elem in Corpus['text']]))+'\n\n')
    print("########################################################\n\n")
    print('Ngrammisation du corpus \n\n')
    # sys.exit()

    Corpus['text'] = n_grammeur(Corpus['text'], n_grammes)
    # print(f'Taille du corpus après n_grammisation de {n_grammes}-grammes : {sum([len(elem.split()) for elem in Corpus['text']])}')
    

    print("########################################################\n\n")
    

    print('Séparation en train (ce qui nous sert à apprendre le modèle + générer le vocabulaire) et test du corpus (on transforme le test en fontion du train)\n\n')  


    Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['text'],Corpus['label'],test_size=0.25)
    

    Encoder = LabelEncoder()
    Train_Y = Encoder.fit_transform(Train_Y) # tranforme les 'neg' et 'pos' en 0 et 1
    # print(Train_Y)

    Test_Y = Encoder.fit_transform(Test_Y) #
    # print(Test_Y)

    print("########################################################\n\n")
    

    print('Génération du vocabulaire en fonction du train\n\n')


    print(f'Taille du vocabulaire avant minmaxdf : {voca(Train_X)}')
    vocab, Train_X = maxmin_df(Train_X, 0.05, 1)  # On max-min df-ise le corpus + on génère le vocabulaire sous-jacent. Si on veut désactiver, on met en paramtres min_df = 0 max df = 1
    print(f'Taille du vocabulaire après minmaxdf : {len(vocab)} min = 0.05 max = 1')
    # print(vocab)
    # bow = simple_bow(Corpus['text'], vocab) # dans le cas de bow basés sur présence/ absence (mettez une des deux en commentaires )
    
    print("########################################################\n\n")
    
    time.sleep(2) # ?
    print('Tranformation du train et du test en bow en fonction du train\n\n')


    bow_Trains = simple_bow(Train_X, vocab) #dans le cas de bow basés sur fréquence (mettez une des deux en commentaires )

    bow_Tests = simple_bow(Test_X, vocab)
    time.sleep(3)
    # print(f"premier doc train bowisé simplement = {bow_Trains[0]}")
    print(f"premier doc test bowisé simplement = {bow_Tests[0]}")
    bow_Trainf = frequency_bow(Train_X, vocab) #dans le cas de bow basés sur fréquence (mettez une des deux en commentaires )

    bow_Testf = frequency_bow(Test_X, vocab)
    # print(f"premier doc train bowisé fréquence = {bow_Trainf[0]}")
    print(f"premier doc test bowisé fréquence = {bow_Testf[0]}")

    bow_Traintfidf = tfidf_bow(Train_X, Train_X, vocab) #dans le cas de bow basés sur tfidf (mettez une des deux en commentaires )

    bow_Testtfidf = tfidf_bow(Test_X, Train_X, vocab)
    #print(f"premier doc train bowisé tf-idf = {bow_Traintfidf[0]}")
    print(f"premier doc test bowisé tf-idf = {bow_Testtfidf[0]}")

    time.sleep(3)
    print("########################################################\n\n")

    print("Chargement de l'algorithme")
   
    Naive = naive_bayes.MultinomialNB()
    print(Naive)

    print("########################################################\n\n")

    print("Chargement de bow simple")
    time.sleep(3)
    Naive.fit(bow_Trains,Train_Y)# predict the labels on validation dataset
    predictions_NB = Naive.predict(bow_Tests)# Use accuracy_score function to get the accuracy
    #print(predictions_NB)
    print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)

    print(classification_report(Test_Y,predictions_NB, labels=[0,1], target_names=['neg','pos']))
    print("########################################################\n\n")

    
    print("Chargement de bow fréquence")
    time.sleep(3)
    Naive.fit(bow_Trainf,Train_Y)# predict the labels on validation dataset
    predictions_NB = Naive.predict(bow_Testf)# Use accuracy_score function to get the accuracy
    # print(predictions_NB)
    print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)

    print(classification_report(Test_Y,predictions_NB, labels=[0,1], target_names=['neg','pos']))
    print("########################################################\n\n")

    print("Chargement de bow tf-idf")
    time.sleep(3)
    Naive.fit(bow_Traintfidf,Train_Y)# predict the labels on validation dataset
    predictions_NB = Naive.predict(bow_Testtfidf)# Use accuracy_score function to get the accuracy
    #print(predictions_NB)
    print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)

    print(classification_report(Test_Y,predictions_NB, labels=[0,1], target_names=['neg','pos']))

if __name__ == '__main__':

    out = '../data/dataset3.csv'


    pipeline(out, n_grammes = 1) # si je  ne précise rien, ce sera des bi-grammes. Ici je précise que je veux des unigrammes


    