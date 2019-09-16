'''
Created on Sep 13, 2019

@author: falensaa
'''

import sys, random, dynet
import numpy as np
import logging

from nparser.trans.task import NNTransParsingTask
from repr.sentence import BiLSTReprBuilder
from tools import neural, utils, datatypes

from nparser.analysis.tools import DummyFeatOutputCache
from nparser.features import FeatOutputCache
from nparser.graph.decoder import FirstOrderDecoder
import nparser.graph.features as gfeatures

class BiLSTReprBuilderWithDrop(BiLSTReprBuilder):
    
    def __init__(self, tokBuilder, lstmDim, lstmLayers, noise=None, lstmDropout=None):
        super().__init__(tokBuilder, lstmDim, lstmLayers, noise, lstmDropout)
    
    def applyLSTMsWithDrop(self, embeds, isTraining, dropWord, noInterruptVec):
        assert type(dropWord) == int
        
        toRunThrough = [ ]
        if dropWord != -1:
            toRunThrough.append(embeds.rootV)
            lstmOff = 1
        else:
            lstmOff = 0
        
        for i, embed in enumerate(embeds.wordsV):
            if dropWord != i:
                toRunThrough.append(embed)
        
        lstmLayers = self._buildLSTMLayers(toRunThrough, isTraining)
        
        wordVs = [ ]
        reprPos = 0
        for i in range(len(embeds.wordsV)):
            if dropWord == i:
                wordVs.append(noInterruptVec.wordsV[i]) # for feature vectors
            else:
                wordVs.append([ lstm[reprPos + lstmOff] for lstm in lstmLayers ])
                reprPos += 1
            
        if dropWord == -1:
            rootV =  noInterruptVec.rootV
        else:
            rootV =  [ lstm[0] for lstm in lstmLayers ]
            
        return neural.VectorManager(wordVs, rootV)
            
            
class NNTransParsingTaskWithDrop(NNTransParsingTask):
    
    def __init__(self, tsystem, anoracle, network, featReprBuilder, dropContextFeat, reprBuilder):
        super().__init__(tsystem, anoracle, network, featReprBuilder)
        
        self.__dropContextFeat = dropContextFeat
        self.__reprBuilder = reprBuilder
        
    def buildLosses(self, vectorsIn, instance, currentEpoch, predictTrain = True):
        result = [ ]
       
        state = self.getTransitionSystem().initialState(len(instance.sentence))
        
        cache = DummyFeatOutputCache()
        
        beforeLSTMs = self.__reprBuilder.prepareEmbeddingsForLSTM(instance, isTraining=True)
        
        vectorsCache = { }
        
        # this is calculated twice, maybe it is possible to just use vectorsIn?
        vectorsCache[None] = self.__reprBuilder.applyLSTMToEmbeds(beforeLSTMs, isTraining=True)
        
        while not self.getTransitionSystem().isFinal(state):
            idToDrop = self.__getDropFeatId(state)
            if idToDrop not in vectorsCache:
                vectorsCache[idToDrop] = self.__reprBuilder.applyLSTMsWithDrop(beforeLSTMs, isTraining=True, dropWord = idToDrop, noInterruptVec =  vectorsCache[None])
                
            iterVectors = vectorsCache[idToDrop]
            
            featReprs = self.getFeatReprBuilder().extractAllAndBuildFeatRepr(state, cache, iterVectors, isTraining=True)
            netOut = self.getNetwork().buildOutput(dynet.esum(featReprs), isTraining=True)
            correctIds = self.getOracle().nextCorrectTransitions(state, instance.correctTree)
            
            # filters
            isValid = lambda x : self.getTransitionSystem().isValidTransition(state, x)
            isCorrect = lambda x : x in correctIds
            
            netVal = netOut.value()
            scoredTransIds = np.argsort(netVal)[::-1]
            
            predictedId = utils.first(lambda x : not isCorrect(x) and isValid(x), scoredTransIds)
            correctId = utils.first(isCorrect, scoredTransIds)
                
            correctVal = netVal[correctId]
            predictedVal = netVal[predictedId] if predictedId != None else -np.inf
            
            nextTransition = correctId
            if predictedId != None:
                mainLoss = self.getNetwork().buildLoss(netOut, correctId, predictedId)
                result.append(mainLoss)
                
                if self.getOracle().handlesExploration() and self.getOracle().doExploration(correctVal, predictedVal, currentEpoch):
                    nextTransition = predictedId
       
            self.getTransitionSystem().applyTransition(state, nextTransition)
            
        # no dropping while predicting train, just to make it faster
        if predictTrain:
            predState = self.getTransitionSystem().initialState(len(instance.sentence))
            self._continueUntilFinal(vectorsIn, predState, FeatOutputCache())
            predictedTree = predState.arcs.buildTree()
        else:
            predictedTree = None
             
        return result, predictedTree
    
    def predict(self, instance, vectorsIn):
        state = self.getTransitionSystem().initialState(len(instance.sentence))
        cache = DummyFeatOutputCache()
        
        beforeLSTMs = self.__reprBuilder.prepareEmbeddingsForLSTM(instance, isTraining=False)
        
        vectorsCache = { }
        vectorsCache[None] = self.__reprBuilder.applyLSTMToEmbeds(beforeLSTMs, isTraining=False)
        while not self.getTransitionSystem().isFinal(state):
            idToDrop = self.__getDropFeatId(state)
            if idToDrop not in vectorsCache:
                vectorsCache[idToDrop] = self.__reprBuilder.applyLSTMsWithDrop(beforeLSTMs, isTraining=False, dropWord = idToDrop, noInterruptVec =  vectorsCache[None])
                
            iterVectors = vectorsCache[idToDrop]
                 
            featReprs = self.getFeatReprBuilder().extractAllAndBuildFeatRepr(state, cache, iterVectors, isTraining=False)
            netOut = self.getNetwork().buildOutput(dynet.esum(featReprs), isTraining=False)
            
            scoredTransIds = np.argsort(netOut.value())[::-1]
            bestTransId = utils.first(lambda x : self.getTransitionSystem().isValidTransition(state, x), scoredTransIds)
            
            if bestTransId == None:
                msg = "No valid transition in state: %s" % str(state)
                self.__logger.error(msg)
                raise RuntimeError(msg)
            
            self.getTransitionSystem().applyTransition(state, bestTransId)
            
        return state.arcs.buildTree()

    def __getDropFeatId(self, state):
        toDrop = None
        if "s0" in self.__dropContextFeat:
            confElem = state.stack.top()
        elif "s1" in self.__dropContextFeat:
            confElem = state.getStackElem(1)
        elif "s2" in self.__dropContextFeat:
            confElem = state.getStackElem(2)
        elif "s3" in self.__dropContextFeat:
            confElem = state.getStackElem(3)
        elif "b0" in self.__dropContextFeat:
            confElem = state.getBufferElem(0)
        elif "b1" in self.__dropContextFeat:
            confElem = state.getBufferElem(1)
        elif "b2" in self.__dropContextFeat:
            confElem = state.getBufferElem(2)
        elif "b3" in self.__dropContextFeat:
            confElem = state.getBufferElem(3)
        elif "b4" in self.__dropContextFeat:
            confElem = state.getBufferElem(4)
        else:
            print("Unknown role", self.__dropContextFeat)
            sys.exit()
        
        # the configuration does not have this role
        if confElem == None:
            return None
        
        # just the element, not children
        if self.__dropContextFeat in [ "s0", "s1", "s2", "b0", "b1", "b2", "b3", "b4" ]:
            return confElem
        
        elC = sorted(state.arcs.getChildren(confElem))
        if len(elC) == 0:
            return None
        
        # left most child
        if self.__dropContextFeat.endswith("L"):
            if elC[0] > confElem:
                return None
            else:
                toDrop = elC[0]

        # right most child
        elif self.__dropContextFeat.endswith("R"):
            if elC[-1] < confElem:
                return None
            else:
                toDrop = elC[-1]
        
        # left child which is not the leftmost
        elif self.__dropContextFeat.endswith("L_"):
            if len(elC) < 2 or elC[1] > confElem:
                return None
            else:
                allL = [ ]
                for c in elC[1:]:
                    if c > confElem:
                        break
                    
                    allL.append(c)
                
                # there is more -- select only one
                toDrop = random.choice(allL)
        
        # right child which is not the rightmost
        elif self.__dropContextFeat.endswith("R_"):
            if len(elC) < 2 or elC[-2] < confElem:
                return None
            else:
                allR = [ ]
                for c in elC[::-1][1:]:
                    if c < confElem:
                        break
                    
                    allR.append(c)
                
                # there is more -- select only one
                toDrop = random.choice(allR)
       
        else:
            print("Unknown role", self.__dropContextFeat)
            sys.exit()

        return toDrop

class FirstOrderDecoderWithDrop(FirstOrderDecoder):
    
    def __init__(self, featReprBuilder, dropContextFeat, reprBuilder):
        super().__init__(featReprBuilder)
        self.__logger = logging.getLogger(self.__class__.__name__)
        
        self.__dropContextFeat = dropContextFeat
        self.__reprBuilder = reprBuilder
        
    def calculateScores(self, instance, vectorsIn, network, scores, isTraining):
        
        # this is only for analysis purposes, so we assume there are correct trees
        if not hasattr(instance, 'property'):
            instance.correctTree = datatypes.sentence2Tree(instance.sentence)
        
        beforeLSTMs = self.__reprBuilder.prepareEmbeddingsForLSTM(instance, isTraining=isTraining)
        
        vectorsCache = { }
        vectorsCache[None] = self.__reprBuilder.applyLSTMToEmbeds(beforeLSTMs, isTraining = isTraining)
        
        hCache = { }
        dCache = { }
        
        for hId in range(-1, len(instance.sentence)):
            for dId in range(-1, len(instance.sentence)):
                toDrop = self.__getDropFeatId(hId, dId, instance.correctTree)
                if toDrop not in vectorsCache:
                    vectorsCache[toDrop] = self.__reprBuilder.applyLSTMsWithDrop(beforeLSTMs, isTraining=isTraining, dropWord = toDrop, noInterruptVec =  vectorsCache[None]) 
                    
                iterVectors = vectorsCache[toDrop]
                
                if (dId, toDrop) in dCache:
                    dRepr, dNr =  dCache[(dId, toDrop)]
                else:
                    depReprs = self.getFeatReprBuilder().extractAndBuildFeatRepr(gfeatures.FeatId.DEP, dId, instance.sentence, iterVectors, isTraining)
                    dRepr = dynet.esum(depReprs) if len(depReprs) > 0 else None
                    dNr = len(depReprs)
                    dCache[(dId, toDrop)] = (dRepr, dNr)
                
                if (hId, toDrop) in hCache:
                    hRepr, hNr = hCache[(hId, toDrop)]
                else:
                    headReprs = self.getFeatReprBuilder().extractAndBuildFeatRepr(gfeatures.FeatId.HEAD, hId, instance.sentence, iterVectors, isTraining)
                    hRepr = dynet.esum(headReprs) if len(headReprs) > 0 else None
                    hNr = len(headReprs)
                    hCache[(hId, toDrop)] = (hRepr, hNr)
                
                #dRepr, dNr = ( depRepr, len(depReprs))
                distRepr = self.getFeatReprBuilder().onlyBuildFeatRepr(gfeatures.FeatId.DIST, (hId, dId), isTraining)
                
                featRepr = [ hRepr, dRepr ]
                featReprNr = hNr + dNr
                if distRepr != None:
                    featRepr.append(distRepr)
                    featReprNr += 1
                
                assert featReprNr == self.getFeatReprBuilder().getNrOfFeatures()
                featRepr = dynet.esum([ f for f in featRepr if f is not None])
                netOut = network.buildOutput(featRepr, isTraining=isTraining)
                
                scores.addOutput(hId, dId, netOut)
                scores.addScore(hId, dId, netOut.scalar_value())
                      
    def __getDropFeatId(self, hId, dId, tree):
        children = sorted(tree.getChildren(dId))
        childrenD = { }
        for c in children:
            childrenD[c] = 1
            
        if hId != -1:
            grandPos = tree.getHead(hId)
        else:
            grandPos = None
    
        sibl = [ ]
        siblD = { }
        for child in tree.getChildren(hId):
            if child != dId:
                sibl.append(child)
                siblD[child] = 1
                
            
        def isNotStruct(wId):
            return wId != hId and wId != dId and wId != grandPos and wId not in childrenD and wId not in siblD
        
        if self.__dropContextFeat == "h":
            toDrop = hId
        elif self.__dropContextFeat == "d":
            toDrop = dId
        elif self.__dropContextFeat == "s":
            if sibl:
                toDrop = random.choice(sibl)
            else:
                toDrop = None
        elif self.__dropContextFeat == "g":
            toDrop = grandPos
        elif self.__dropContextFeat == "c":
            if children:
                toDrop = random.choice(children)
            else:
                toDrop = None
                
        elif self.__dropContextFeat == "h+-1_":
            opts = [ ]
            
            if hId > 0 and isNotStruct(hId - 1):
                opts.append(hId - 1)
                
            if hId+1 < tree.nrOfTokens() and isNotStruct(hId + 1):
                opts.append(hId + 1)
                
            if opts:
                toDrop = random.choice(opts)
            else:
                toDrop = None
                
        elif self.__dropContextFeat == "d+-1_":
            opts = [ ]
            
            if dId > 0 and isNotStruct(dId - 1):
                opts.append(dId - 1)
                
            if dId+1 < tree.nrOfTokens() and isNotStruct(dId + 1):
                opts.append(dId + 1)
                
            if opts:
                toDrop = random.choice(opts)
            else:
                toDrop = None
                
                
        elif self.__dropContextFeat == "h+-2_":
            opts = [ ]
            
            if hId > 1 and isNotStruct(hId - 2):
                opts.append(hId - 2)
                
            if hId+2 < tree.nrOfTokens() and isNotStruct(hId + 2):
                opts.append(hId + 2)
                
            if opts:
                toDrop = random.choice(opts)
            else:
                toDrop = None
                
                
        elif self.__dropContextFeat == "d+-2_":
            opts = [ ]
            
            if dId > 1 and isNotStruct(dId - 2):
                opts.append(dId - 2)
                
            if dId+2 < tree.nrOfTokens() and isNotStruct(dId + 2):
                opts.append(dId + 2)
                
            if opts:
                toDrop = random.choice(opts)
            else:
                toDrop = None
                
        elif self.__dropContextFeat == "d+-3_":
            opts = [ ]
            
            if dId > 2 and isNotStruct(dId - 3):
                opts.append(dId - 3)
                
            if dId+3 < tree.nrOfTokens() and isNotStruct(dId + 3):
                opts.append(dId + 3)
                
            if opts:
                toDrop = random.choice(opts)
            else:
                toDrop = None
                
        else:
            self.__logger.warn("Unknown feat", self.__dropContextFeat)
            sys.exit()
        
        return toDrop
