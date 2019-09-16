'''
Created on Jun 29, 2018

@author: falensaa
'''

import dynet
import logging
import abc

from tools.neural import TrainInstance, VectorManager
  
class SentenceReprBuilder(object):
    __metaclass__ = abc.ABCMeta
    
    ##
    # word2i operations
    
    @abc.abstractmethod
    def readData(self, sentences):
        pass
    
    @abc.abstractmethod    
    def save(self, pickleOut):
        pass
    
    @abc.abstractmethod
    def load(self, pickleIn):
        pass

    ##
    # instance operations
    
    @abc.abstractmethod
    def buildInstance(self, sentence):
        pass
    
    @abc.abstractmethod
    def initializeParameters(self, model):
        pass
    
    ##
    # vector operations
        
    @abc.abstractmethod
    def getDim(self):
        pass
    
    @abc.abstractmethod
    def prepareVectors(self, sentInstance, isTraining):
        pass
    
    
################################################################
# SentenceReprBuilders
################################################################

class CollectReprBuilder(SentenceReprBuilder):
    
    def __init__(self, builders):
        self.__logger =  logging.getLogger(self.__class__.__name__)
        self.__builders = builders
         
    ##
    # word2i operations
     
    def readData(self, sentences):
        for sent in sentences:
            for tok in sent:
                for bld in self.__builders:
                    bld.addToken(tok)
        
        self.__logger.debug("Features read: %s" % ", ".join([bld.getFeatInfo() + "/" + str(bld.getDim()) for bld in self.__builders]))
    
    def save(self, pickleOut):
        for rb in self.__builders:
            rb.save(pickleOut)
    
    def load(self, pickleIn):
        for rb in self.__builders:
            rb.load(pickleIn)
    
    ##
    # instance operations    
    
    def initializeParameters(self, model):
        for rb in self.__builders:
            rb.initializeParameters(model)
    
    def buildInstance(self, sentence):
        return TrainInstance(sentence,
                             [ [ rb.buildInstance(tok) for tok in sentence ] for rb in self.__builders ])
    
    ##
    # vector operations
    
    def getDim(self):
        result = sum([rb.getDim() for rb in self.__builders ])
        return result
    
    def prepareVectors(self, sentInstance, isTraining):
        rootV = self.__getRootVector()
        wordV = self.__prepareWordVectors(sentInstance, isTraining)
        return VectorManager(wordV, rootV)
    
    def __prepareWordVectors(self, sentInstance, isTraining):
        wordV = [ ]
        for tokInstance in zip(*sentInstance.ids):
            bldVectors = [ ]
            for bldInst, bld in zip(tokInstance, self.__builders):
                bldVectors.append(bld.getTokenVector(bldInst, isTraining))
            wordV.append(dynet.concatenate(bldVectors))
        
        return wordV

    def __getRootVector(self):
        return dynet.concatenate([rb.getRootVector() for rb in self.__builders])


class BiLSTReprBuilder(SentenceReprBuilder):
    
    def __init__(self, tokBuilder, lstmDim, lstmLayers, noise=None, lstmDropout=None):
        # global
        self.__logger =  logging.getLogger(self.__class__.__name__)
        
        self.__tokBuilder = tokBuilder
        self.__lstmDim = lstmDim
        self.__noise = noise
        self.__dropout = lstmDropout
        
        # lstms
        assert type(lstmLayers) == type(1)
        self.__lstmLayers = lstmLayers
        
        self.__forwardLstms = [ None ] * self.__lstmLayers
        self.__backwardLstms = [ None ] * self.__lstmLayers
        
    ##
    # word2i operations
    def readData(self, sentences):
        self.__tokBuilder.readData(sentences)
    
    def save(self, pickleOut):
        self.__tokBuilder.save(pickleOut)
    
    def load(self, pickleIn):
        self.__tokBuilder.load(pickleIn)
       
    ##
    # instance operations
    def initializeParameters(self, model):
        self.__logger.debug("Initialize: (%i, %i)" % (self.__tokBuilder.getDim(), 2 * self.__lstmDim))
        
        self.__tokBuilder.initializeParameters(model)
        
        # first layer of lstms
        self.__forwardLstms[0] = dynet.VanillaLSTMBuilder(1, self.__tokBuilder.getDim(), self.__lstmDim, model)
        self.__backwardLstms[0] = dynet.VanillaLSTMBuilder(1, self.__tokBuilder.getDim(), self.__lstmDim, model)
        
        # other layers
        for i in range(1, self.__lstmLayers):    
            self.__forwardLstms[i] = dynet.VanillaLSTMBuilder(1, 2 * self.__lstmDim, self.__lstmDim, model)
            self.__backwardLstms[i] = dynet.VanillaLSTMBuilder(1, 2 * self.__lstmDim, self.__lstmDim, model)
        
    def buildInstance(self, sentence):
        return self.__tokBuilder.buildInstance(sentence)
    
    ##
    # vector operations
    def getDim(self):
        return 2 * self.__lstmDim
    
    def prepareVectors(self, instance, isTraining):
        embeds = self.prepareEmbeddingsForLSTM(instance, isTraining)
        return self.applyLSTMToEmbeds(embeds, isTraining)
    
    def prepareEmbeddingsForLSTM(self, instance, isTraining):
        vectors = self.__tokBuilder.prepareVectors(instance, isTraining)
        rootRepr = self.__applyNoise(vectors.rootV, isTraining)
        wordRepr = [ self.__applyNoise(v, isTraining) for v in vectors.wordsV ]
        return VectorManager(wordRepr, rootRepr)
    
    def applyLSTMToEmbeds(self, embeds, isTraining):
        inputReprs = [ embeds.rootV ] +  embeds.wordsV
        lstmLayers = self._buildLSTMLayers(inputReprs, isTraining)
        
        wordVs = [ ]
        for i in range(len(inputReprs) - 1):
            wordVs.append([ lstm[i+1] for lstm in lstmLayers ])
            
        rootV =  [ lstm[0] for lstm in lstmLayers ]
            
        result = VectorManager(wordVs, rootV)
        return result
    
    def _buildLSTMLayers(self, inputs, isTraining):
        self.__setDropout(isTraining)
        
        # first layer:
        lstmLayers = [ None ] * self.__lstmLayers
        lstmLayers[0] = self.__runThroughBiLstm(inputs, self.__forwardLstms[0], self.__backwardLstms[0])
        
        for i in range(1, self.__lstmLayers):
            lstmLayers[i] = self.__runThroughBiLstm(lstmLayers[i-1], self.__forwardLstms[i], self.__backwardLstms[i])
            
        return lstmLayers 
    
    def __runThroughBiLstm(self, vectors, forwardLstm, backwardLstm):
        forwardInit = forwardLstm.initial_state()
        backwardInit = backwardLstm.initial_state()
        
        forward = [ x.output() for x in forwardInit.add_inputs(vectors) ]
        backward = [x.output() for x in backwardInit.add_inputs(reversed(vectors)) ]

        return [ dynet.concatenate([fw, bw]) for (fw, bw) in zip(forward, reversed(backward)) ]
    
    def __applyNoise(self, exp, train):
        if self.__noise == None or not train:
            return exp
        
        return dynet.noise(exp, self.__noise)
    
    def __setDropout(self, isTraining):
        if not self.__dropout:
            return
        
        lstms = self.__forwardLstms + self.__backwardLstms
        for lstm in lstms:
            lstm.set_dropout(self.__dropout) if isTraining else lstm.disable_dropout()
