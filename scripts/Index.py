
import glob

import pprint

class Document(object) : #création de la classe document

    def __init__(self,fic,lg='fr'): 
        """
        création du constructeur de la classe document. Il prend en paramètres de création un nom de fichier et une langue, par défaut 'fr'.
            :params -> self = lui-même. Un objet peut faire référence à lui-même, à ses méthodes à ses attributs : self.attribut et self.methode()
            :params -> fic = str
            :params -> lg = str
       """
        self.name = fic # on crée l'attribut name qui prend la valeur de fic

        self.langue = lg #idem pour l'attribut langue. 

        self.content = open(fic,encoding='utf-8').read() # on crée l'attribut content qui est le contenu du fichier

    def tokenized(self) :

        self.tokens = self.content.split() # on crée l'attribut tokens qui représente la liste des tokens du contenu du fichier

    def lemmatized(self) : # on crée l'attribut lemmas qui représente la liste des tokens lemmatisés du contenu du fichier
        pass

    def postaggued(self) : #idem
        pass

    def lemma_pos(self) : #on crée un attribut associant le lemme à sa part-of-speech
        pass

class Fable(Document) :
    """
        création de la classe Fable qui HERITE de toutes les propriétés de Document
    """

    def __init__(self,fic,lg='fr'):

        Document.__init__(self,fic,lg)  # le constructeur de Fable() est le même que celui de Document

        self.titre = self.content.split('\n')[0] # On rajoute au constructeur de Fable() l'attribut titre qui représente la première ligne de son contenu. 


# class Corpus(object) :


class Index(object) : 

    """
        Création de la classe Index()
    """

    def __init__(self,docs):

        """
        COnstructeur de l'index qui prend en entrée le paramètre docs qui est une liste d'objets de la classe Document() (ou de ses filles telle que Fable()) 
        :params -> docs = list[Document()]

        """

        self.dic = {}
        self.empty = True
        if type(docs) != list or len(docs) <2 :
            raise ValueError("Ce n'est pas une liste ou bien elle ne contient qu'un doc") # On vérifie que c'est vraiment une liste ou qu'il y'a plus d'un document dans la liste

        for doc in docs : # si les docs n'ont pas l'attribut titre, on le rajoute automatiquement. L'index est agnostique au type de document, il ne sait pas si c'est un Document() ou une Fable()

            if not hasattr(doc,'titre') :

               doc.titre = doc.name
            doc.tokenized() # on tokénize chaque doc
        self.docs = docs # on crée l'attribut docs qui s'approprie les docs (tokénisés)


    def maxFreq(self,word):
        """
        lorsque l'index est construit, pour un mot donné, on veut connaître le document dans lequel il apparait le plus
        algo : 

            rappel -> index = dic{mot:{doc:[occurrences,tf-idf]}}
            on crée une variable maxi = 0
            on crée une variable maxdoc = ""
            pour chaque doc associé à ce mot : 
                on regarde son occurrence. Si elle est supérieure au max, maxdoc = doc, si elle est égale, maxdoc += ','+doc, sinon on fait rien

        """

        pass 


    def initialize(self):

        if self.empty :

            for doc in self.docs :
                Index.indexing(self,doc,doc.tokens)
            self.empty = False
            # pprint.pprint(self.dic)

        else :
            raise AttributeError("L'index n'est pas vide, vous ne pouvez pas l'initialiser")

    def blockIndex(self,) :

        self.dic = {elem: [(cle, v[cle]) for cle in v] for (elem, v) in self.dic.items()}
        self.blocked = True

    def unBlockIndex(self):


        self.dic = {elem:{tup[0]:tup[1] for tup in v} for (elem, v) in self.dic.items()}
        self.blocked = False


    def indexing(self,doc,tokens):
        """ 
        CF page wikipédia sur la programmation récursive. 

        """
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





if __name__ == '__main__':

    fables = [Fable(doc) for doc in glob.glob('fables/*')]

    index = Index(fables)

    index.initialize()
    new_doc = Fable(glob.glob('new_doc/*.txt')[0])
    new_doc.tokenized()
    index.indexing(new_doc,new_doc.tokens)
    index.blockIndex()
    index.unBlockIndex()
    pprint.pprint(index.dic)
    max  = index.maxFreq('courage')
    # index.initialize()



