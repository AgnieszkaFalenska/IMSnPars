'''
Created on 23.08.2017

@author: falensaa
'''

import logging
import numpy as np

from tools import neural

class NNGraphParsingTask(neural.NNTreeTask):
    def __init__(self, mstAlg, featReprBuilder, decod, network, augmentScore):
        self.__mst = mstAlg
        self.__featReprBuilder = featReprBuilder
        self.__decoder = decod
        self.__network = network
        self.__augmentScore = augmentScore
        self.__logger = logging.getLogger(self.__class__.__name__)
    
    def getMSTAlg(self):
        return self.__mst
    
    def getDecoder(self):
        return self.__decoder
    
    def getFeatReprBuilder(self):
        return self.__featReprBuilder
    
    def getNetwork(self):
        return self.__network
        
    def handlesNonProjectiveTrees(self):
        return True
    
    def initializeParameters(self, model, reprDim):
        self.__featReprBuilder.initializeParameters(model, reprDim)
        self.__network.initializeParameters(model, reprDim, 1)
             
    def renewNetwork(self):
        self.__network.renewNetwork()
        
    def buildLosses(self, vectors, instance, currentEpoch, predictTrain = True):
        scores = self.__mst.emptyScores(instance) 
        self.__decoder.calculateScores(instance, vectors, self.__network, scores, isTraining=True)
        
        # decoding twice to get the augmented tree
        if predictTrain:
            predictTrainTree = self.__mst.findMSTTree(scores)
        else:
            predictTrainTree = None
    
        if self.__augmentScore:
            self.__decoder.augmentScores(scores, instance)
            predictTree = self.__mst.findMSTTree(scores)
        elif predictTrain:
            predictTree = predictTrainTree
        else:
            predictTree = self.__mst.findMSTTree(scores)
    
        errors = self.__decoder.findErrors(scores, instance.correctTree, predictTree)
        losses = self.__network.buildLosses(errors)
            
        return losses, predictTrainTree
    
    def predict(self, instance, vectors):
        scores = self.__mst.emptyScores(instance) 
        self.__decoder.calculateScores(instance, vectors, self.__network, scores, isTraining=False)
        predictTree = self.__mst.findMSTTree(scores)
        return predictTree
    
    
####################
# with labels together with arcs
####################

class NNGraphParsingTaskWithLbl(neural.NNTreeTask):
    def __init__(self, mstAlg, featReprBuilder, decod, network, augmentScore, lblDict):
        self.__mst = mstAlg
        self.__featReprBuilder = featReprBuilder
        self.__decoder = decod
        self.__network = network
        self.__augmentScore = augmentScore
        self.__lblDict = lblDict
        self.__logger = logging.getLogger(self.__class__.__name__)
    
    def getLblDict(self):
        return self.__lblDict
    
    def handlesNonProjectiveTrees(self):
        return True
    
    def initializeParameters(self, model, reprDim):
        self.__featReprBuilder.initializeParameters(model, reprDim)
        self.__network.initializeParameters(model, reprDim, self.__lblDict.getNrOfLbls())
             
    def renewNetwork(self):
        self.__network.renewNetwork()
        
    def buildLosses(self, vectors, instance, currentEpoch, predictTrain = True):    
        scores = self.__mst.emptyScores(instance)
        self.__decoder.calculateScores(instance, vectors, self.__network, scores, isTraining=True)
        
        # decoding twice to get the augmented tree
        if predictTrain:
            predictTrainTree = self.__predictTree(scores)
        else:
            predictTrainTree = None

        if self.__augmentScore:
            self.__decoder.augmentScores(scores, instance)
            predictTree = self.__predictTree(scores)
        elif predictTrain:
            predictTree = predictTrainTree
        else:
            predictTree = self.__predictTree(scores)
            
        errors = self.__findErrors(scores, instance.correctTree, predictTree)
        losses = self.__network.buildLosses(errors) 
        return losses, predictTrainTree
    
    def __findErrors(self, outputs, correct, predicted):
        errs = [ ]
        
        for tPos in range(correct.nrOfTokens()):
            if correct.getHead(tPos) == predicted.getHead(tPos) and correct.getLabel(tPos) == predicted.getHead(tPos):
                continue
            
            corrHead = correct.getHead(tPos)
            corrLblId = self.__lblDict.getLblId(correct.getLabel(tPos))
            predHead = predicted.getHead(tPos)
            predLblId = self.__lblDict.getLblId(predicted.getLabel(tPos))
            
            errs.append((outputs.getOutput(predHead, tPos)[predLblId], outputs.getOutput(corrHead, tPos)[corrLblId]))
           
        return errs
    
    def __predictTree(self, scores):
        tree = self.__mst.findMSTTree(scores)
        
        labels = [ ]
        for dId in range(tree.nrOfTokens()):
            hId = tree.getHead(dId)
            
            output = scores.getOutput(hId, dId)
            maxDim = np.argmax(output.value())
            
            lbl = self.__lblDict.getLbl(maxDim)
            labels.append(lbl)
        
        tree.setLabels(labels)
        return tree
        
    def predict(self, instance, vectors):
        scores = self.__mst.emptyScores(instance) 
        self.__decoder.calculateScores(instance, vectors, self.__network, scores, isTraining=False)
        predictTree = self.__predictTree(scores)
        return predictTree