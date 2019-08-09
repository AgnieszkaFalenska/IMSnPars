'''
Created on Jun 29, 2018

@author: falensaa
'''

import dynet
import logging
import abc
import pickle
import random

from tools.nn_tools import TrainInstance, VectorManager

class TokenReprBuilder(abc.ABC):
    
    ##
    # word2i dictionaries operations
    
    @abc.abstractmethod
    def addToken(self, token):
        pass
    
    @abc.abstractmethod
    def save(self, pickleOut):
        pass
     
    @abc.abstractmethod
    def load(self, pickleIn):
        pass
    
    @abc.abstractmethod
    def getFeatInfo(self):
        pass
    
    
    ##
    # instance operations
    
    @abc.abstractmethod
    def initializeParameters(self, model):
        pass
    
    @abc.abstractmethod
    def buildInstance(self, token):
        pass
    
    ##
    # vector operations
        
    @abc.abstractmethod
    def getDim(self):
        pass
        
    @abc.abstractmethod
    def getTokenVector(self, tokInstance, isTraining):
        pass
    
    @abc.abstractmethod
    def getRootVector(self):
        pass
    

################################################################
# TokenReprBuilders
################################################################

class WordReprBuilder(TokenReprBuilder):
    def __init__(self, dim, wordDropout):
        self.__dim = dim
        self.__vocab = { }
        self.__wordsFreq = { }
        self.__wordDropout = wordDropout
        self.__logger =  logging.getLogger(self.__class__.__name__)
        
        # additional entries - root, unknown
        self.__addEntries = 2
        self.__lookup = None

    ##
    # word2i operations
    
    def addToken(self, token):        
        norm = token.getNorm()
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
        return "Words: %i" % len(self.__vocab)
    
    ##
    # instance opeations
    
    def initializeParameters(self, model):
        self.__lookup = model.add_lookup_parameters((len(self.__vocab) + self.__addEntries, self.__dim))
        
    def buildInstance(self, token):
        return self.__vocab.get(token.getNorm())
    
    ##
    # vector operations
    
    def getDim(self):
        return self.__dim
    
    def getTokenVector(self, wordId, isTraining):
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
    
    def __wordIdWithDropout(self, wordId):
        if self.__wordDropout == None or wordId == None:
            return wordId
        
        dropProb = self.__wordDropout / ( self.__wordDropout + self.__wordsFreq.get(wordId))
        if random.random() < dropProb:
            return None
        
        return wordId
    
class POSReprBuilder(TokenReprBuilder):
    def __init__(self, dim):
        self.__logger =  logging.getLogger(self.__class__.__name__)
        self.__dim = dim
        self.__pos = { }
        
        # additional entries - root, unknown
        self.__addEntries = 2
        self.__lookup = None

    ##
    # word2i operations
    
    def addToken(self, token):        
        posId = self.__pos.get(token.pos, None)
        if posId == None:
            posId = len(self.__pos)
            self.__pos[token.pos] = posId
    
    def save(self, pickleOut):
        pickle.dump(self.__pos, pickleOut)
    
    def load(self, pickleIn):
        self.__pos = pickle.load(pickleIn)
    
    def getFeatInfo(self):
        return "POS: %i" % len(self.__pos)
    
    ##
    # instance opeations
    def initializeParameters(self, model):
        self.__lookup = model.add_lookup_parameters((len(self.__pos) + self.__addEntries, self.__dim))

    def buildInstance(self, token):
        return self.__pos.get(token.pos)
    
    ##
    # vector operations
    
    def getDim(self):
        return self.__dim
    
    def getTokenVector(self, posId, _):
        if posId == None:
            return self.__getUnknVector()
        else:
            return self.__lookup[posId]
    
    def getRootVector(self):
        return self.__lookup[len(self.__pos) + 1]
    
    def __getUnknVector(self):
        return self.__lookup[len(self.__pos)]


class MorphReprBuilder(TokenReprBuilder):
    def __init__(self, dim):
        self.__dim = dim
        self.__morph = { }
        
        # additional entries - root, unknown
        self.__addEntries = 2
        self.__lookup = None

    ##
    # word2i operations
    def addToken(self, token):  
        morphId = self.__morph.get(token.morph, None)
        if morphId == None:
            morphId = len(self.__morph)
            self.__morph[token.morph] = morphId
    
    def save(self, pickleOut):
        pickle.dump(self.__morph, pickleOut)
    
    def load(self, pickleIn):
        self.__morph = pickle.load(pickleIn)
    
    def getFeatInfo(self):
        return "Morph: %i" % len(self.__morph)
    
    ##
    # instance opeations
    def initializeParameters(self, model):
        self.__lookup = model.add_lookup_parameters((len(self.__morph) + self.__addEntries, self.__dim))

    def buildInstance(self, token):
        return self.__morph.get(token.morph)
    
    ##
    # vector operations
    
    def getDim(self):
        return self.__dim
    
    def getTokenVector(self, morphId, _):
        if morphId == None:
            return self.__getUnknVector()
        else:
            return self.__lookup[morphId]
    
    def getRootVector(self):
        return self.__lookup[len(self.__morph) + 1]
    
    def __getUnknVector(self):
        return self.__lookup[len(self.__morph)]


class CharLstmReprBuilder(TokenReprBuilder):
    def __init__(self, dim, lstmDim, charDropout=None, lstmDropout=None):
        self.__dim = dim
        self.__lstmDim = lstmDim
        self.__chars = { }
        self.__charFreq = { }
        
        # additional entries - unknown, <w>, </w>
        self.__addEntries = 3
        self.__lookup = None
        
        self.__forwardLstm = None
        self.__backwardLstm = None
        self.__rootVec = None
        
        self.__dropout = lstmDropout
        self.__charDropout = charDropout

    ##
    # word2i operations
    
    def addToken(self, token):
        for c in token.orth:
            if c not in self.__chars:
                cId = len(self.__chars)
                self.__chars[c] = cId
            else:
                cId = self.__chars[c]

            if self.__charDropout:
                if cId not in self.__charFreq:
                    self.__charFreq[cId] = 1
                else:
                    self.__charFreq[cId] += 1

    
    def save(self, pickleOut):
        pickle.dump((self.__chars, self.__charFreq), pickleOut)
    
    def load(self, pickleIn):
        self.__chars, self.__charFreq = pickle.load(pickleIn)

    def getFeatInfo(self):
        return "Chars [BiLSTM]: %i" % len(self.__chars)
    
    ##
    # instance opeations
    def initializeParameters(self, model):
        self.__lookup = model.add_lookup_parameters((len(self.__chars) + self.__addEntries, self.__dim))
        self.__rootVec = model.add_parameters((self.getDim()))
        
        self.__forwardLstm = dynet.VanillaLSTMBuilder(1, self.__dim, self.__lstmDim, model)
        self.__backwardLstm = dynet.VanillaLSTMBuilder(1, self.__dim, self.__lstmDim, model)

    def buildInstance(self, token):
        return [ self.__chars.get(c) for c in token.orth ]

    ##
    # vector operations
    
    def getDim(self):
        return 2 * self.__lstmDim
    
    def getTokenVector(self, charIds, isTraining):
        self.__setDropout(isTraining)
            
        if isTraining and self.__charDropout:
            charIds = [ self.__charIdWithDropout(cId) for cId in charIds ]

        charVecs = [ self.__getBegVector() ]
        charVecs += [ self.__lookup[cId] if cId != None else self.__getUnknCVector() for cId in charIds  ]
        charVecs.append( self.__getEndVector() )
            
        forwardInit = self.__forwardLstm.initial_state()
        backwardInit = self.__backwardLstm.initial_state()
        
        result = [ ]
        result.append(forwardInit.add_inputs(charVecs)[-1].output())
        result.append(backwardInit.add_inputs(reversed(charVecs))[-1].output())
            
        return dynet.concatenate(result)

    def getRootVector(self):
        return self.__rootVec.expr()
    
    def __getBegVector(self):
        return self.__lookup[len(self.__chars)]
    
    def __getEndVector(self):
        return self.__lookup[len(self.__chars) + 1]
    
    def __getUnknCVector(self):
        return self.__lookup[len(self.__chars) + 2]
    
    def __setDropout(self, isTraining):
        if not self.__dropout:
            return 
        
        if isTraining:
            self.__forwardLstm.set_dropout(self.__dropout)
            self.__backwardLstm.set_dropout(self.__dropout)
        else:
            self.__forwardLstm.disable_dropout()
            self.__backwardLstm.disable_dropout()
    
    def __charIdWithDropout(self, cId):
        if self.__charDropout == None or cId == None:
            return cId

        dropProb = self.__charDropout / ( self.__charDropout + self.__charFreq.get(cId))
        if random.random() < dropProb:
            return None

        return cId
