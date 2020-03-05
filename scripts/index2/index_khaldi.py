import spacy
import pandas as pd
import math

nlp = spacy.load("fr_core_news_sm")

text = "la bouffe est pas mauvaise. Si je le trouve pas bon. J'avoue il est vraiment mauvais"

doc = nlp(text)

for token in doc:
    print(token.text, token.lemma_, token.pos_, token.is_stop)

cols = ("Word", "Lemma", "POS", "Explain", "Stopword")
rows = []


for t in doc:
    row = [t.text, t.lemma_, t.pos_, spacy.explain(t.pos_), t.is_stop]
    rows.append(row)

#print(rows)

df = pd.DataFrame(rows, columns=cols)
df

word = df["Word"]
print(word)

wordict = dict.fromkeys(word, 0)
wordict

for w in word:
    wordict[w] += 1

wordict

pd.DataFrame([wordict])


def calculTF(wordict, w):
    tfDict = {}
    wCount = len(w)
    for word, count in wordict.items():
        tfDict[word] = count/float(wCount)
    return tfDict


tf = calculTF(wordict, w)
tf

def calculIDF(docList):
    idfDict = {}
    N = len(docList)

    idfDict = dict.fromkeys(docList[0].keys(), 0)
    for doc in docList:
        for word, val in doc.items():
            if val > 0:
                idfDict[word] += 1

    for word, val in idfDict.items():
        idfDict[word] = math.log10(N / float(val))

    return idfDict

idfs = calculIDF([wordict])

def calculTFIDF(tf, idfs):
    tfidf = {}
    for word, val in tf.items():
        tfidf[word] = val*idfs[word]
    return tfidf

tfidf = calculTFIDF(tf, idfs)

pd.DataFrame([tfidf])


