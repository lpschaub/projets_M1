from collections import Counter
import spacy 
from spacy import displacy
from math import exp, expml
nlp = spacy.load('fr_core_news_sm')

index_word = Counter()
index_doc = Counter()

def word_count (doc, index_word, index_doc):
    sentence = nlp(doc)
    current_index_word = Counter()
    for token in sentence:
        if token.is_stop == False:
            current_index_word[token.lemma_]+=1
            
    for word, count in current_index_word.items():
        index_word[iword]+=count
        index_doc[word]+=1
        
    return len(sentence)

doc1 = "la nourriture est pas mauvaise"
doc2 = "si, je le trouve bon"
doc3 = "j'avoue il est vraiment mauvais"

list_doc = [doc1, doc2,  doc3]

words_total = 0 
for doc in list_doc:
    words_total += word_count(doc, index_word, index_doc)

most_freq = str(index_word.most_common(1))

print(index_word)
print("The most frequent word is " + most_freq + ".")

for word, value in index_word.items():
    tf = int(value) / words_total 
    print(f"TF of the word {word} is equal to {tf:.2f}.")
    idf = log2(len(list_doc) * (1 / index_doc[word]))
    print(f"IDF of the word {word} is equal to {idf:.2f}.")
