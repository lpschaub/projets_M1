


class Evaluation(object) :

    def __init__(self,reference,predit):

        self.__vrais_positifs = 0
        self.__faux_positifs = 0
        self.__faux_negatifs = 0
        self.values = ('neg','pos')

        if len(reference) != len(predit) :
            raise AttributeError("Il y'a un pb dans vos données")

        for i in range(len(reference)):
            if reference[i] == predit[i]  ==  self.values[1] :
                self.__vrais_positifs += 1
            elif reference[i] != predit[i]:
                if reference[i] == self.values[1] :
                    self.__faux_negatifs += 1
                else :
                    self.__faux_positifs += 1

    def precision(self):

        return self.__vrais_positifs/(self.__vrais_positifs+self.__faux_positifs)

    def rappel(self):

        return self.__vrais_positifs / (self.__vrais_positifs + self.__faux_negatifs)

    def f_mesure(self):

        return 2*((self.precision()*self.rappel())/(self.precision()+self.rappel()))

    def getVraisPos(self):

        return self.__vrais_positifs

    def getFauxNeg(self):

        return self.__faux_negatifs

    def getFauxPos(self):

        return self.__faux_positifs

if __name__ == '__main__':

    # ref = ['0','1','0','1','0','0']
    # pred = ['0', '0', '1', '1', '0', '0']

    eval = Evaluation(ref,pred)

    print(eval.getVraisPos())
    print(eval.getFauxNeg())
    print(eval.getFauxPos())
    print(eval.f_mesure())
