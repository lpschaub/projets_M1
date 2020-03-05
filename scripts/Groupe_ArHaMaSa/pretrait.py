from glob import glob
import os
from Evaluation import Evaluation
from Predict import Predict
predicted = []
expected = []


def getFile(fic,pol): 
    x = 0
    for f in glob(fic+'/*') : 
        x += 1
        os.system(f'mv {f} {pol}/')
        if x == 500 : 

            break


def getcontentlabel(file) :
    
    expect = '' 
    if 'pos' in file : 
        expect = 'pos'
    else : 
        expect = 'neg'

    return open(file).read(),expect


def prediction(file,expect) : 


    pred = Predict(file)
    pred.predict()
    predicted.append(pred.predicted)
    expected.append(expect)


if __name__ == '__main__':


    for fic in glob('../corpus/imdb/*/*.txt') : 
        print(fic)
        file, label = getcontentlabel(fic)
        print(file)
        print(label)

        prediction(file,label)


    print(predicted)
    print(expected)

    eval = Evaluation(expected,predicted)

    
    print(eval.getVraisPos())
    print(eval.getFauxNeg())
    print(eval.getFauxPos())
    print(eval.f_mesure())