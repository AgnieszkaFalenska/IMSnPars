'''
Created on 2 lip 2014

@author: agnieszka
'''

import pickle

from repr import word

class SuperTagReprBuilder(word.TokenReprBuilder):
    def __init__(self, dim):
        self.__dim = dim
        self.__stags = { }
        
        # additional entries - root, unknown
        self.__addEntries = 2
        self.__lookup = None

    ##
    # word2i operations
    
    def addToken(self, token):
        stagId = self.__stags.get(token.getSuperTag(), None)
        if stagId == None:
            stagId = len(self.__stags)
            self.__stags[token.getSuperTag()] = stagId
    
    def save(self, pickleOut):
        pickle.dump(self.__stags, pickleOut)
    
    def load(self, pickleIn):
        self.__stags = pickle.load(pickleIn)
    
    def getFeatInfo(self):
        return "Stags: %i" % len(self.__stags)
    
    ##
    # instance opeations
    def initializeParameters(self, model):
        self.__lookup = model.add_lookup_parameters((len(self.__stags) + self.__addEntries, self.__dim))

    def buildInstance(self, token):
        return self.__stags.get(token.getSuperTag())
    
    ##
    # vector operations
    
    def getDim(self):
        return self.__dim
    
    def getTokenVector(self, stagId, _):
        if stagId == None:
            return self.__getUnknVector()
        else:
            return self.__lookup[stagId]
    
    def getRootVector(self):
        return self.__lookup[len(self.__stags) + 1]
    
    def __getUnknVector(self):
        return self.__lookup[len(self.__stags)]
    

class SuperTagger(object):
    '''Model creates tags from heads and directions of dependents'''
    
    def __init__(self):
        self.possibTags = set()
        
    def annotate(self, tokens):
        deps = { }
        for t in tokens:
            headPos = t.stackHead
            if headPos not in deps:
                deps[headPos] = [ ]
                
            depRel = self.__getDepRel(t.getTokPos(), headPos)
            deps[headPos].append(depRel)
            
        for t in tokens:
            superTag = self.__mainSuperTag(t)
            
            if t.getTokPos() in deps:
                superTag += "+" + "_".join(sorted(list(set(deps[t.getTokPos()]))))
                
            t.supertag = superTag
            self.possibTags.add(superTag)
            
    def annotateSents(self, sentences):
        for sent in sentences:
            self.annotate(sent)
        
    def __getHeadRel(self, tokPos, headPos):
        if headPos > tokPos:
            return "R"
        elif headPos < tokPos:
            return "L"
        else:
            print("Problem, the same tokPos and headPos:", tokPos, headPos)
            return None

    def __getDepRel(self, tokPos, headPos):
        if headPos < tokPos:
            return "R"
        elif headPos > tokPos:
            return "L"
        else:
            print("Problem, the same tokPos and headPos:", tokPos, headPos)
            return None
    
    def __mainSuperTag(self, token):
        result = token.stackDep
        result += "/" + self.__getHeadRel(token.getTokPos(), token.stackHead)
        return result