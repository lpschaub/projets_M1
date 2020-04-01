import csv, spacy, nltk 
from spacy import displacy
from nltk.util import bigrams, trigrams
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

total_words = ""
with open('mini_csv.csv') as file:
  reader = csv.reader(file, delimiter='\t')
  for row in reader:
      total_words += row[0]

vocab = set()
vocab_lemm = set()
without_stop_words =set()
nlp = spacy.load('en_core_web_sm')
doc = nlp(total_words)
for token in doc:
    vocab.add(token.text)
    if token.is_punct == False:
        vocab_lemm.add(token.lemma_)
        if token.is_stop == False:
            without_stop_words.add(token.text)

print(f"LEN Vocabulaire en mode normal: {str(len(vocab))}.\nLEN Vocabulaire lemmatisé: {str(len(vocab_lemm))}.\nLEN Vocabulaire sans stop-words: {str(len(without_stop_words))}.\n")

total_words = total_words.split()
bigramms = bigrams(total_words)
set_bigr = set()
for bigramm in bigramms:
    set_bigr.add(bigramm)
trigramms = trigrams(total_words)
set_trigr = set()
for trigramm in trigramms:
    set_trigr.add(trigramm)

print(f"LEN Vocabulaire bigrammes: {len(set_bigr)}.\nLEN Vocabulaire trigrammes: {len(set_trigr)}." )

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(total_words)
feature_names = vectorizer.get_feature_names()
dense = vectors.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=feature_names)

print(f"\nTF-IDF: {df}")