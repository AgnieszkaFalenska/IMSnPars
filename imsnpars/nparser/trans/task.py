'''
Created on 23.08.2017

@author: falensaa
'''

import dynet
import numpy as np

import nparser.features
from tools import utils
from tools.neural import NNTreeTask
    
class NNTransParsingTask(NNTreeTask):
    def __init__(self, tsystem, anoracle, network, featReprBuilder):
        self.__oracle = anoracle
        self.__tsystem = tsystem
        self.__network = network
        self.__featReprBuilder = featReprBuilder
        
    def initializeParameters(self, model, reprDim):
        self.__featReprBuilder.initializeParameters(model, reprDim)
        self.__network.initializeParameters(model, reprDim, self.__tsystem.getNrOfTransitions())
             
    def getTransLabeler(self):
        return self.__tsystem
    
    def renewNetwork(self):
        self.__network.renewNetwork()
                           
    def handlesNonProjectiveTrees(self):
        return self.__oracle.handlesNonProjectiveTrees()
    
    def buildLosses(self, vectors, instance, currentEpoch, predictTrain):
        result = [ ]
       
        state = self.__tsystem.initialState(len(instance.sentence))
        
        cache = nparser.features.FeatOutputCache()
        while not self.__tsystem.isFinal(state):
            featReprs = self.__featReprBuilder.extractAllAndBuildFeatRepr(state, cache, vectors, isTraining=True)
            
            netOut = self.__network.buildOutput(dynet.esum(featReprs), isTraining=True)
            
            correctIds = self.__oracle.nextCorrectTransitions(state, instance.correctTree)
            
            # filters
            isValid = lambda x : self.__tsystem.isValidTransition(state, x)
            isCorrect = lambda x : x in correctIds
            
            netVal = netOut.value()
            scoredTransIds = np.argsort(netVal)[::-1]
            
            predictedId = utils.first(lambda x : not isCorrect(x) and isValid(x), scoredTransIds)
            correctId = utils.first(isCorrect, scoredTransIds)
                
            correctVal = netVal[correctId]
            predictedVal = netVal[predictedId] if predictedId != None else -np.inf
            
            nextTransition = correctId
            if predictedId != None:
                mainLoss = self.__network.buildLoss(netOut, correctId, predictedId)
                if mainLoss is not None:
                    result.append(mainLoss)
                
                if self.__oracle.handlesExploration() and self.__oracle.doExploration(correctVal, predictedVal, currentEpoch):
                    nextTransition = predictedId

            self.__tsystem.applyTransition(state, nextTransition)
            
        if predictTrain:
            predState = self.__tsystem.initialState(len(instance.sentence))
            self.__continueUntilFinal(vectors, predState, cache)
            predictedTree = predState.arcs.buildTree()
        else:
            predictedTree = None
             
        return result, predictedTree
    
    def predict(self, instance, vectors):
        state = self.__tsystem.initialState(len(instance.sentence))
        
        cache = nparser.features.FeatOutputCache()
        self.__continueUntilFinal(vectors, state, cache)
        return state.arcs.buildTree()
    
    def __continueUntilFinal(self, vectors, state, cache):
        while not self.__tsystem.isFinal(state):
            featReprs = self.__featReprBuilder.extractAllAndBuildFeatRepr(state, cache, vectors, isTraining=False)
            netOut = self.__network.buildOutput(dynet.esum(featReprs), isTraining=False)
            
            scoredTransIds = np.argsort(netOut.value())[::-1]
            bestTransId = utils.first(lambda x : self.__tsystem.isValidTransition(state, x), scoredTransIds)
            
            if bestTransId == None:
                msg = "No valid transition in state: %s" % str(state)
                self.__logger.error(msg)
                raise RuntimeError(msg)
            
            self.__tsystem.applyTransition(state, bestTransId)
    
