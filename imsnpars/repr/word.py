'''
Created on Jun 29, 2018

@author: falensaa
'''

import dynet
import logging
import abc
import pickle
import random

from tools import utils

class TokenReprBuilder(object):
    __metaclass__ = abc.ABCMeta
    
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
    
class ExtEmbWordReprBuilder(TokenReprBuilder):
    def __init__(self, dim, wordDropout, embFilename, embUpdate, useNorm):
        self.__logger =  logging.getLogger(self.__class__.__name__)
        
        self.__dim = dim
        self.__vocab = { }
        self.__wordsFreq = { }
        self.__wordDropout = wordDropout
        self.__embFilename = embFilename
        self.__embUpdate = embUpdate
        self.__extEmbeddings = None
        self.__useNorm = useNorm
        
        # additional entries - root, unknown
        self.__addEntries = 2
        self.__lookup = None

    ##
    # word2i operations
    
    def addToken(self, token):
        info = self.__getTokenInfo(token)
        self.__addToVocab(info)

    def save(self, pickleOut):
        pickle.dump((self.__vocab, self.__wordsFreq), pickleOut)
    
    def load(self, pickleIn):
        self.__vocab, self.__wordsFreq = pickle.load(pickleIn)
    
    def getFeatInfo(self):
        return "EWords: %i" % len(self.__vocab)
    
    ##
    # instance operations
    
    def initializeParameters(self, model):
        if not self.__embUpdate and self.__embFilename:
            assert not self.__extEmbeddings 
            self.__addOOVWordsToDict()
        
        self.__lookup = model.add_lookup_parameters((len(self.__vocab) + self.__addEntries, self.__dim))
        self.__logger.info("Setting update: %s", str(self.__embUpdate))
        self.__lookup.set_updated(self.__embUpdate)
            
    def buildInstance(self, token):
        info = self.__getTokenInfo(token)
        return self.__vocab.get(info)
    
    ##
    # vector operations
    
    def getDim(self):
        return self.__dim
    
    def getTokenVector(self, wordId, isTraining):
        # TODO: ugly place to put this but works with "noUpdate" option
        if not self.__extEmbeddings and isTraining:
            self.__readEmbeddings()
            self.__loadEmbIntoLookup()
        
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
    
    def __readEmbeddings(self):
        self.__logger.info("Reading embeddings from file: %s" % self.__embFilename)
        self.__extEmbeddings = { }
        for line in utils.smartOpen(self.__embFilename):
            line = utils.decode(line)
            parts = line.split()
            assert len(parts) == self.__dim + 1
            
            try:
                wVec = [ float(v) for v in parts[1:]]
                self.__extEmbeddings[parts[0]] = wVec
            except:
                self.__logger.warn("Could not build a vector from %s" % line)
                
    def __loadEmbIntoLookup(self):
        allPretrained = 0
        for word in self.__extEmbeddings:
            wordId = self.__vocab.get(word)
            if wordId != None:
                self.__lookup.init_row(wordId, self.__extEmbeddings[word])
                allPretrained += 1
        
        self.__logger.info("Loaded embeddings into lookup for %i out of %i (%i)" % (allPretrained, len(self.__vocab), len(self.__extEmbeddings)))
            
    def __addOOVWordsToDict(self):
        allNew = 0
        for line in utils.smartOpen(self.__embFilename):
            line = utils.decode(line)
            parts = line.split()
            assert len(parts) == self.__dim + 1
            
            wordId = self.__vocab.get(parts[0])
            if wordId == None:
                self.__addToVocab(parts[0])
                allNew += 1
    
        self.__logger.info("Added %i OOV words, total dictionary %i" % (allNew, len(self.__vocab)))
        
    def __addToVocab(self, info, addFreq=True):
        wId = self.__vocab.get(info, None)
        
        if wId == None:
            wId = len(self.__vocab)
            self.__vocab[info] = wId
    
        if self.__wordDropout:
            if wId not in self.__wordsFreq:
                self.__wordsFreq[wId] = 1
            elif addFreq:
                self.__wordsFreq[wId] += 1
                
    def __getTokenInfo(self, token):
        return token.norm if self.__useNorm else token.orth 
                
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
        return self.__lookup[len(self.__pos)]
    
    def __getUnknVector(self):
        return self.__lookup[len(self.__pos) + 1]


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
    def __init__(self, dim, lstmDim, charDropout=None, lstmDropout=None, useBorders=False):
        self.__dim = dim
        self.__lstmDim = lstmDim
        self.__chars = { }
        self.__charFreq = { }
        
        # additional entries - unknown, <w>, </w>
        self.__addEntries = 3 if useBorders else 1
        self.__useBorders = useBorders
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

        charVecs = [ self.__lookup[cId] if cId != None else self.__getUnknCVector() for cId in charIds  ]
        charVecs = self.__addBorders(charVecs)
            
        forwardInit = self.__forwardLstm.initial_state()
        backwardInit = self.__backwardLstm.initial_state()
        
        result = [ ]
        result.append(forwardInit.add_inputs(charVecs)[-1].output())
        result.append(backwardInit.add_inputs(reversed(charVecs))[-1].output())
            
        return dynet.concatenate(result)

    def __addBorders(self, charVecs):
        if not self.__useBorders:
            return charVecs
        
        return [ self.__getBegVector() ] + charVecs + [ self.__getEndVector() ]
    
    def getRootVector(self):
        return self.__rootVec.expr()
    
    def __getBegVector(self):
        return self.__lookup[len(self.__chars) + 1]
    
    def __getEndVector(self):
        return self.__lookup[len(self.__chars) + 2]
    
    def __getUnknCVector(self):
        return self.__lookup[len(self.__chars)]
    
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
