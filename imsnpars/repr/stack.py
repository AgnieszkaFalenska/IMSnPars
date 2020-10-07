'''
Created on Sep 27, 2019

@author: falensaa
'''

import logging
import pickle
import random

from repr import word

class StackHeadReprBuilder(word.TokenReprBuilder):
    def __init__(self, dim, wordDropout):
        self.__dim = dim
        self.__vocab = { }
        self.__wordsFreq = { }
        self.__wordDropout = wordDropout
        self.__logger =  logging.getLogger(self.__class__.__name__)
        
        # additional entries - root, unknown, grandroot
        self.__addEntries = 3
        self.__lookup = None

    ##
    # word2i operations
    
    def addToken(self, token):        
        norm = token.headNorm
        if norm is None:
            return
        
        wId = self.__vocab.get(norm, None)
        
        if wId == None:
            wId = len(self.__vocab)
            self.__vocab[norm] = wId
    
        if self.__wordDropout:
            if wId not in self.__wordsFreq:
                self.__wordsFreq[wId] = 1
            else:
                self.__wordsFreq[wId] += 1

    def save(self, pickleOut):
        pickle.dump((self.__vocab, self.__wordsFreq), pickleOut)
    
    def load(self, pickleIn):
        self.__vocab, self.__wordsFreq = pickle.load(pickleIn)
    
    def getFeatInfo(self):
        return "StackHeads: %i" % len(self.__vocab)
    
    ##
    # instance opeations
    
    def initializeParameters(self, model):
        self.__lookup = model.add_lookup_parameters((len(self.__vocab) + self.__addEntries, self.__dim))
        
    def buildInstance(self, token):
        headNorm = token.headNorm
        if headNorm is None:
            return -1
        
        return self.__vocab.get(headNorm)
    
    ##
    # vector operations
    
    def getDim(self):
        return self.__dim
    
    def getTokenVector(self, wordId, isTraining):
        if wordId == -1:
            return self.__getGrandRootVector()
        
        if isTraining:
            wordId = self.__wordIdWithDropout(wordId)
            
        if wordId == None:
            return self.__getUnknVector()
        else:
            return self.__lookup[wordId]
    
    def getRootVector(self):
        return self.__lookup[len(self.__vocab) + 1]
    
    def __getUnknVector(self):
        return self.__lookup[len(self.__vocab)]
    
    def __getGrandRootVector(self):
        return self.__lookup[len(self.__vocab) + 2]
    
    def __wordIdWithDropout(self, wordId):
        if self.__wordDropout == None or wordId == None:
            return wordId
        
        dropProb = self.__wordDropout / ( self.__wordDropout + self.__wordsFreq.get(wordId))
        if random.random() < dropProb:
            return None
        
        return wordId
    
class StackLblReprBuilder(word.TokenReprBuilder):
    def __init__(self, dim):
        self.__logger =  logging.getLogger(self.__class__.__name__)
        self.__dim = dim
        self.__lbl = { }
        
        # additional entries - root, unknown
        self.__addEntries = 2
        self.__lookup = None

    ##
    # word2i operations
    
    def addToken(self, token):        
        lblId = self.__lbl.get(token.stackDep, None)
        if lblId == None:
            lblId = len(self.__lbl)
            self.__lbl[token.stackDep] = lblId
    
    def save(self, pickleOut):
        pickle.dump(self.__lbl, pickleOut)
    
    def load(self, pickleIn):
        self.__lbl = pickle.load(pickleIn)
    
    def getFeatInfo(self):
        return "StackLbl: %i" % len(self.__lbl)
    
    ##
    # instance opeations
    def initializeParameters(self, model):
        self.__lookup = model.add_lookup_parameters((len(self.__lbl) + self.__addEntries, self.__dim))

    def buildInstance(self, token):
        return self.__lbl.get(token.stackDep)
    
    ##
    # vector operations
    
    def getDim(self):
        return self.__dim
    
    def getTokenVector(self, lblId, _):
        if lblId == None:
            return self.__getUnknVector()
        else:
            return self.__lookup[lblId]
    
    def getRootVector(self):
        return self.__lookup[len(self.__lbl)]
    
    def __getUnknVector(self):
        return self.__lookup[len(self.__lbl) + 1]

    