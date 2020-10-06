'''
Created on 23.08.2017

@author: falensaa
'''

import pickle
import numpy as np
import dynet

import nparser.features
from tools import  utils, neural
from nparser.graph import features as gfeatures

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

class LabelerGraphTask(neural.NNTreeTask):
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
    
    def buildLosses(self, vectors, instance, currentEpoch, predictTrain):
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
            
            depRepr = self.__featReprBuilder.extractAndBuildFeatRepr(gfeatures.FeatId.DEP, dId, instance.sentence, vectors, isTraining)
            headRepr = self.__featReprBuilder.extractAndBuildFeatRepr(gfeatures.FeatId.HEAD, hId, instance.sentence, vectors, isTraining)
            distRepr = self.__featReprBuilder.onlyBuildFeatRepr(gfeatures.FeatId.DIST, (hId, dId), isTraining)
                
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
     
    
class DummyLabeler(neural.NNTreeTask):
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
        
    def buildLosses(self, vectors, instance, currentEpoch, predictTrain):
        return [ ], None
    
    def predict(self, instance, tree, vectors):
        return None
    
    def reportLabels(self):
        return False