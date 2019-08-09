'''
Created on 23.08.2017

@author: falensaa
'''

import pickle
import numpy as np
import dynet

from tools import  utils
from tools.nn_tools import NNTreeTask
from nparser.graph.gp_features import FeatId
from nparser.pars_features import FeatOutputCache

class LblTagDict(object):
    def __init__(self):
        self.__lbl2Id = { }
        self.__id2lbl = { }
        
    def getNrOfLbls(self):
        return len(self.__lbl2Id)
    
    def getLbl(self, lId):
        return self.__id2lbl[lId]
    
    def getLblId(self, lbl):
        return self.__lbl2Id[lbl]
    
    def readData(self, sentences):
        for sent in sentences:
            for tok in sent:
                if tok.dep not in self.__lbl2Id:
                    self.__id2lbl[len(self.__lbl2Id)] = tok.dep
                    self.__lbl2Id[tok.dep] = len(self.__lbl2Id)
                    
    def save(self, pickleOut):
        pickle.dump((self.__lbl2Id, self.__id2lbl), pickleOut)
    
    def load(self, pickleIn):
        (self.__lbl2Id, self.__id2lbl) = pickle.load(pickleIn)

class LabelerGraphTask(NNTreeTask):
    def __init__(self, featReprBuilder, network, reprLayer):
        self.__network = network
        self.__featReprBuilder = featReprBuilder
        self.__lbls = LblTagDict()
        self.__reprLayer = reprLayer
        
    def save(self, pickleOut):
        self.__lbls.save(pickleOut)

    def load(self, pickleIn):
        self.__lbls.load(pickleIn)
        
    def readData(self, sentences):
        self.__lbls.readData(sentences)
    
    def initializeParameters(self, model, reprDim):
        self.__featReprBuilder.initializeParameters(model, reprDim)
        self.__network.initializeParameters(model, reprDim, self.__lbls.getNrOfLbls())
        
    def renewNetwork(self):
        self.__network.renewNetwork()
    
    def buildLosses(self, vectors, instance, currentEpoch, predictTrain = True):
        outputsLbls = self.__buildLblOutputs(instance, instance.correctTree, vectors, isTraining=True)
        correctLbls = [ self.__lbls.getLblId(instance.correctTree.getLabel(tokPos)) for tokPos in range(instance.correctTree.nrOfTokens()) ]
        losses = self.__buildBestLosses(outputsLbls, correctLbls)
            
        if predictTrain:
            lblPred = self.__predictLbls(outputsLbls)
        else:
            lblPred = None
            
        return losses, lblPred
                
    def predict(self, instance, tree, vectors):
        transLblOut = self.__buildLblOutputs(instance, tree, vectors, isTraining=False)
        transLbls = [ ]
        for output in transLblOut:
            netVal = output.value()
            bestTagId = np.argmax(netVal)
            transLbls.append(self.__lbls.getLbl(bestTagId))
        
        return transLbls
       
    def reportLabels(self):
        return True
    
    def __buildLblOutputs(self, instance, tree, vectors, isTraining):
        outputs = [ ]
        for dId in range(tree.nrOfTokens()):
            hId = tree.getHead(dId)
            
            depRepr = self.__featReprBuilder.extractAndBuildFeatRepr(FeatId.DEP, dId, instance.sentence, vectors, isTraining)
            headRepr = self.__featReprBuilder.extractAndBuildFeatRepr(FeatId.HEAD, hId, instance.sentence, vectors, isTraining)
            distRepr = self.__featReprBuilder.onlyBuildFeatRepr(FeatId.DIST, (hId, dId), isTraining)
                
            featRepr = headRepr + depRepr
            if distRepr != None:
                featRepr.append(distRepr)
                
            assert len(featRepr) == self.__featReprBuilder.getNrOfFeatures()
            featRepr = dynet.esum(featRepr)
            netOut = self.__network.buildOutput(featRepr, isTraining=isTraining)
            outputs.append(netOut)
                 
        return outputs
     
    def __predictLbls(self, lblOuts):
        predLbls = [ ]
        for output in lblOuts:
            netVal = output.value()
            bestTagId = np.argmax(netVal)
            bestLbl = self.__lbls.getLbl(bestTagId)
            predLbls.append(bestLbl)
        return predLbls
    
    def __buildBestLosses(self, outputsLbls, correctLbls):
        losses = [ ]
        for (output, correct) in zip(outputsLbls, correctLbls):
            sortedIds = np.argsort(output.value())[::-1]
            predicted = utils.first(lambda x : x != correct, sortedIds)
            losses.append(self.__network.buildLoss(output, correct, predicted))
        return losses
     
    
class LabelerTransTask(NNTreeTask):
    def __init__(self, tsystem, anoracle, noLblOracle, network, featReprBuilder):
        self.__oracle = anoracle
        self.__noLblOracle = noLblOracle
        self.__tsystem = tsystem
        self.__network = network
        self.__featReprBuilder = featReprBuilder
        
    def initializeParameters(self, model, reprDim):
        self.__featReprBuilder.initializeParameters(model, reprDim)
        self.__network.initializeParameters(model, reprDim, self.__tsystem.getNrOfTransitions())
             
    def getTransLabeler(self):
        return self.__tsystem
    
    def getNetwork(self):
        return self.__network
    
    def getFeatReprBuilder(self):
        return self.__featReprBuilder
            
    def readData(self, sentences):
        self.__tsystem.readData(sentences)
        
    def save(self, pickleOut):
        self.__tsystem.save(pickleOut)
         
    def load(self, pickleIn):
        self.__tsystem.load(pickleIn)
        
    def renewNetwork(self):
        self.__network.renewNetwork()
                                       
    def handlesNonProjectiveTrees(self):
        return self.__oracle.handlesNonProjectiveTrees()
    
    def buildLosses(self, vectors, instance, currentEpoch, predictTrain = True):
        result = [ ]
        
        state = self.__tsystem.initialState(len(instance.sentence))
        cache = FeatOutputCache()
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
             
        return result, predictedTree.getLabels()
    
    def predict(self, instance, tree, vectors):
        state = self.__tsystem.initialState(len(instance.sentence))
        cache = FeatOutputCache()
        while not self.__tsystem.isFinal(state):
            featReprs = self.__featReprBuilder.extractAllAndBuildFeatRepr(state, cache, vectors, isTraining=False)
            netOut = self.__network.buildOutput(dynet.esum(featReprs), isTraining=False)
            correctIds = self.__noLblOracle.nextCorrectTransitions(state, tree)
            
            isCorrect = lambda x : self.__tsystem.getSysTrans(x) in correctIds
            isValid = lambda x : self.__tsystem.isValidTransition(state, x)
            
            scoredTransIds = np.argsort(netOut.value())[::-1]
            bestTransId = utils.first(lambda x : isValid(x) and isCorrect(x), scoredTransIds)
            
            if bestTransId == None:
                msg = "No valid transition in state: %s" % str(state)
                self.__logger.error(msg)
                raise RuntimeError(msg)
                
            self.__tsystem.applyTransition(state, bestTransId)
            
        assert state.arcs.buildTree().buildStrNoLabels() == tree.buildStrNoLabels()
        return state.arcs.buildTree().getLabels()
       
    def reportLabels(self):
        return True 

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
            
class DummyLabeler(NNTreeTask):
    def __init__(self, lblData = None):
        self.__lblData = lblData
        
    def save(self, pickleOut):
        if self.__lblData is not None:
            self.__lblData.save(pickleOut)
    
    def load(self, pickleIn):
        if self.__lblData is not None:
            self.__lblData.load(pickleIn)

    def readData(self, sentences):
        if self.__lblData is not None:
            self.__lblData.readData(sentences)
        
    def initializeParameters(self, model, reprDim):
        pass
    
    def renewNetwork(self):
        pass
        
    def buildLosses(self, vectors, instance, currentEpoch, predictTrain = False):
        return [ ], None
    
    def predict(self, instance, tree, vectors):
        return None
    
    def reportLabels(self):
        return False