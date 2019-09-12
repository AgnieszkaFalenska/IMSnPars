'''
Created on 23.08.2017

@author: falensaa
'''

import dynet
import logging
import numpy as np

from tools import neural

class ParserNetwork(neural.NNetwork):
    def __init__(self, mlpHiddenDim, nonLinFun, featIds):
        self.__logger = logging.getLogger(self.__class__.__name__)
        
        # dimentionalities
        self.__mlpHiddenDim = mlpHiddenDim
         
        # network parameters
        self.__paramsMlpHidLayers = { }
        self.__paramsMlpHiddenBias = None
        self.__paramsMlpOutputW = None
        self.__paramsMlpOutputBias = None
         
        # network layers
        self.__hidLayers = { }

        self.__hiddenBias = None
        self.__outputW = None
        self.__outputBias = None
        
        self.__nonLinFun = nonLinFun
        
        for featId in featIds:
            self.__paramsMlpHidLayers[featId] = None
            self.__hidLayers[featId] = None
        
    # NNetwork method
    def initializeParameters(self, model, inputDim, outDim):
        # MLP parameters
        for featId in self.__paramsMlpHidLayers:
            self.__paramsMlpHidLayers[featId] = model.add_parameters((self.__mlpHiddenDim, inputDim))

        self.__logger.debug("Initialize: (%i, %i)" % (outDim, self.__mlpHiddenDim))
        self.__paramsMlpHiddenBias = model.add_parameters(self.__mlpHiddenDim)
         
        # one hidden layer
        self.__paramsMlpOutputW = model.add_parameters((outDim, self.__mlpHiddenDim))
        self.__paramsMlpOutputBias = model.add_parameters((outDim))
        
    # NNetwork method
    def renewNetwork(self):
        for featId in self.__paramsMlpHidLayers:
            self.__hidLayers[featId] = dynet.parameter(self.__paramsMlpHidLayers[featId])

        self.__hiddenBias = dynet.parameter(self.__paramsMlpHiddenBias)
        self.__outputW = dynet.parameter(self.__paramsMlpOutputW)
        self.__outputBias = dynet.parameter(self.__paramsMlpOutputBias)
    
    # NNetwork method
    def buildOutput(self, inputRepr, isTraining):
        hiddenOut = self.__nonLinFun(inputRepr + self.__hiddenBias)
        outLayer = self.__outputW * hiddenOut + self.__outputBias
        return outLayer
    
    def buildFeatOutput(self, featId, featVec, isTraining):
        return self.__hidLayers[featId] * featVec
    
    # NNetwork method
    def buildLoss(self, output, correct, predicted = None):
        if isinstance(predicted, np.int64):
            predicted = predicted.item()
        
        if isinstance(correct, np.int64):
            correct = correct.item()
                    
        if predicted == None:
            return self.__allElemLoss(output, correct)
        else:
            return self.__oneElemLoss(output, correct, predicted)
    
    def buildLosses(self, errors):
        return [ dynet.rectify( 1 + pred - corr) for (pred, corr) in errors ]
    
    def __oneElemLoss(self, output, correct, predicted):
        return dynet.rectify(1 + output[predicted] - output[correct])
    
    def __allElemLoss(self, output, correct):
        return dynet.hinge(output, correct, 1)
