'''
Created on 23.08.2017

@author: falensaa
'''

import dynet
import logging
import numpy as np

from tools import neural

class ParserNetwork(neural.NNetwork):
    def __init__(self, mlpHiddenDim, nonLinFun, featIds, trainMargin):
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
        self.__trainMargin = trainMargin
     
        self.__nonLinFun = nonLinFun
        
        for featId in featIds:
            self.__paramsMlpHidLayers[featId] = None
            self.__hidLayers[featId] = None
        
    # NNetwork method
    def initializeParameters(self, model, inputDim, outDim):
        # MLP parameters
        for featId in self.__paramsMlpHidLayers:
            self.__paramsMlpHidLayers[featId] = model.add_parameters((self.__mlpHiddenDim, inputDim))
 
        self.__logger.debug("Initialize: (%i, %i, %i)" % (inputDim, self.__mlpHiddenDim, outDim))
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
    def buildLoss(self, output, correctId, predictedId = None):
        if isinstance(predictedId, np.int64):
            predictedId = predictedId.item()
        
        if isinstance(correctId, np.int64):
            correctId = correctId.item()
                    
        if predictedId == None:
            return self.__allElemLoss(output, correctId)
        else:
            return self.__oneElemLoss(output[correctId], output[predictedId])
    
    def buildLosses(self, errOutputs):
        result = [ ]
        for (predOut, corrOut) in errOutputs:
            loss = self.__oneElemLoss(corrOut, predOut)
            if loss is not None:
                result.append(loss)
        return result
    
    def __oneElemLoss(self, correctOut, predictedOut):
        if correctOut.value() < predictedOut.value() + self.__trainMargin:
            return predictedOut - correctOut
        
        return None
    
    def __allElemLoss(self, output, correctId):
        return dynet.hinge(output, correctId, self.__trainMargin)
        
        
        
