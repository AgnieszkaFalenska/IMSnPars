'''
Created on 23.08.2017

@author: falensaa
'''

import logging
import numpy as np
import dynet

from tools import neural

class NNGraphParsingTask(neural.NNTreeTask):
    def __init__(self, mstAlg, featReprBuilder, decod, network, augmentScore, imposeOneRoot):
        self.__mst = mstAlg
        self.__featReprBuilder = featReprBuilder
        self.__decoder = decod
        self.__network = network
        self.__augmentScore = augmentScore
        self.__imposeOneRoot = imposeOneRoot
        self.__logger = logging.getLogger(self.__class__.__name__)
    
    def handlesNonProjectiveTrees(self):
        return True
    
    def initializeParameters(self, model, reprDim):
        self.__featReprBuilder.initializeParameters(model, reprDim)
        self.__network.initializeParameters(model, reprDim, 1)
             
    def renewNetwork(self):
        self.__network.renewNetwork()
        
    def predict(self, instance, vectors):
        scores = self.__mst.emptyScores(instance) 
        self.__decoder.calculateScores(instance, vectors, self.__network, scores, isTraining=False)
        predictTree = self.__mst.findMSTTree(scores, imposeOneRoot=self.__imposeOneRoot)
        return predictTree
    
    def buildLosses(self, vectors, instance, currentEpoch, predictTrain):
        scores = self.__mst.emptyScores(instance) 
        self.__decoder.calculateScores(instance, vectors, self.__network, scores, isTraining=True)
        
        # decoding twice to get the augmented tree
        if predictTrain:
            predictTrainTree = self.__mst.findMSTTree(scores, imposeOneRoot=False)
        else:
            predictTrainTree = None
    
        if self.__augmentScore:
            self.__decoder.augmentScores(scores, instance)
            predictTree = self.__mst.findMSTTree(scores, imposeOneRoot=False)
        elif predictTrain:
            predictTree = predictTrainTree
        else:
            predictTree = self.__mst.findMSTTree(scores, imposeOneRoot=False)
    
        errorsOuts = self.__buildErrorOutputs(scores, instance.correctTree, predictTree)
        losses = self.__network.buildLosses(errorsOuts)
            
        return losses, predictTrainTree
    
    def __buildErrorOutputs(self, outputs, correct, predicted):
        errs = [ tPos for tPos in range(correct.nrOfTokens()) if correct.getHead(tPos) != predicted.getHead(tPos) ]
           
        if len(errs) == 0:
            return [ ]
           
        return [ (outputs.getOutput(predicted.getHead(tPos), tPos), outputs.getOutput(correct.getHead(tPos), tPos)) for tPos in errs ]
    
    
    
####################
# with labels together with arcs
####################

class NNGraphParsingTaskWithLbl(neural.NNTreeTask):
    def __init__(self, mstAlg, featReprBuilder, decod, network, augmentScore, lblDict, imposeOneRoot):
        self.__mst = mstAlg
        self.__featReprBuilder = featReprBuilder
        self.__decoder = decod
        self.__network = network
        self.__augmentScore = augmentScore
        self.__lblDict = lblDict
        self.__imposeOneRoot = imposeOneRoot
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
         
    def predict(self, instance, vectors):
        scores = self.__mst.emptyScores(instance) 
        self.__decoder.calculateScores(instance, vectors, self.__network, scores, isTraining=False)
        predictTree = self.__predictTree(scores, isTraining=False)
        return predictTree
    
    def buildLosses(self, vectors, instance, currentEpoch, predictTrain = True):    
        scores = self.__mst.emptyScores(instance)
        self.__decoder.calculateScores(instance, vectors, self.__network, scores, isTraining=True)
        
        # decoding twice to get the augmented tree
        if predictTrain:
            predictTrainTree = self.__predictTree(scores, isTraining=True)
        else:
            predictTrainTree = None
 
        if self.__augmentScore:
            self.__decoder.augmentScores(scores, instance)
            predictTree = self.__predictTree(scores, isTraining=True)
        elif predictTrain:
            predictTree = predictTrainTree
        else:
            predictTree = self.__predictTree(scores, isTraining=True)
             
        errOutputs = self.__buildErrorOutputs(scores, instance.correctTree, predictTree)
        losses = self.__network.buildLosses(errOutputs) 
        return losses, predictTrainTree
     
        
    def __buildErrorOutputs(self, scores, correctTree, predictedTree):
        result = [ ]
        for tPos in range(correctTree.nrOfTokens()):
            corrHead = correctTree.getHead(tPos)
            predHead = predictedTree.getHead(tPos)
        
            corrOutputs = scores.getOutput(corrHead, tPos)
            predOutputs = scores.getOutput(predHead, tPos)
            
            corrLblId = self.__lblDict.getLblId(correctTree.getLabel(tPos))
            predLblId = self.__lblDict.getLblId(predictedTree.getLabel(tPos))
            
            ### tree errors    
            if corrHead != predHead:
                result.append((predOutputs[predLblId], dynet.max_dim(corrOutputs)))
            
            ### lbl errors
            worstLblId = max((scr, lId) for (lId, scr) in enumerate(corrOutputs.value()) if lId != corrLblId)[1]
            result.append((corrOutputs[worstLblId], corrOutputs[corrLblId]))
                         
        return result
            
    def __predictTree(self, scores, isTraining):
        tree = self.__mst.findMSTTree(scores, imposeOneRoot=self.__imposeOneRoot and not isTraining)
         
        labels = [ ]
        for dId in range(tree.nrOfTokens()):
            hId = tree.getHead(dId)
             
            output = scores.getOutput(hId, dId)
            maxDim = np.argmax(output.value())
             
            lbl = self.__lblDict.getLbl(maxDim)
            labels.append(lbl)
         
        tree.setLabels(labels)
        return tree
    
