'''
Created on Sep 26, 2019

@author: falensaa, some code taken from https://github.com/UppsalaNLP/uuparser/blob/elmo/barchybrid/src/elmo.py
'''

import logging
import dynet as dy
import h5py
import numpy as np
from repr import sentence
from tools.neural import VectorManager

class ELMoReprBuilder(sentence.SentenceReprBuilder):
    
    def __init__(self, tokBuilder, elmoFile, elmoGamma, elmoLearnGamma):
        # global
        self.__logger =  logging.getLogger(self.__class__.__name__)
        
        # lower level
        self.__tokBuilder = tokBuilder
        
        # parameters
        self.__sentenceData = h5py.File(elmoFile, 'r')
        
        self.__weights = None
        self.__rootV = None
        self.__gamma = None if elmoLearnGamma else elmoGamma

        self.__numLayers, _, self.__embDim = next(iter(self.__sentenceData.values())).shape
        
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
        self.__tokBuilder.initializeParameters(model)
        
        self.__weights = model.add_parameters(self.__numLayers, name="elmo-layer-weights", init="uniform", scale=1.0)
        self.__rootV = model.add_parameters(self.__embDim, name="elmo-rootV", init="uniform")
         
        if self.__gamma is None:
            self.__gamma = model.add_parameters(1, name="elmo-gamma", init=1.0)
        
    def buildInstance(self, sentence):
        instance = self.__tokBuilder.buildInstance(sentence) 
        
        sentenceText = self.__sentToText(sentence)
        sentenceData = self.__sentenceData.get(sentenceText)
        
        if not sentenceData:
            sentenceData = self.__sentenceData.get(sentenceText.replace("_", " "))
        
            if not sentenceData:
                keys = self.__sentenceData.keys()
                for k in list(keys)[:10]:
                    print(k)
                    
                raise ValueError("The sentence %s could not be found in the ELMo data." % sentenceText)

        instance.elmoSent = sentenceData
        return instance
    
    def __sentToText(self, sentence):
        sentenceText = "\t".join([tok.orth for tok in sentence])
        sentenceText = sentenceText.replace('.', '$period$')
        sentenceText = sentenceText.replace('/', '$backslash$')
        return sentenceText
    
    ##
    # vector operations
    def getDim(self):
        return self.__embDim + self.__tokBuilder.getDim()
    
    def prepareVectors(self, instance, isTraining):
        vectors = self.__tokBuilder.prepareVectors(instance, isTraining)
        
        wordVs = [ ]
        for i, wordV in enumerate(vectors.wordsV):
            wordVs.append(dy.concatenate([wordV, self.__getWordRepr(instance, i)]))
        
        rootV = dy.concatenate([vectors.rootV, self.__rootV])
        return VectorManager(wordVs, rootV)
    
    def __getWordRepr(self, instance, i):
        layers = self.__getSentenceLayers(instance, i)

        normalizedWeights = dy.softmax(self.__weights)
        
        yHat = [ ]
        for layer, weight in zip(layers, normalizedWeights):
            tensor = dy.inputTensor(layer)
            yHat.append(tensor * weight)
            
        # Sum the layer contents together
        return dy.esum(yHat) * self.__gamma
    
    def __getSentenceLayers(self, instance, i):
        """
        Returns the layers for the word at position i in the sentence.
        :param i: Index of the word.
        :return: (n x d) matrix where n is the number of layers
                 and d the number of embedding dimensions.
        """

        # self.sentence_weights is of dimensions (n x w x d)
        # with n as the number of layers
        # w the number of words
        # and d the number of dimensions.

        # Therefore, we must iterate over the matrix to retrieve the layer
        # for each word separately.
        layers = []
        for layer in instance.elmoSent:
            layers.append(layer[i])

        return np.array(layers)
    
