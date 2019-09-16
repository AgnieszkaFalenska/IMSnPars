'''
Created on 10.02.2019

@author: falensaa
'''

import abc
import numpy as np
import dynet
import logging

from tools import datatypes, utils

def _getGradient(embed):
    return np.linalg.norm(embed.gradient(), ord=2)

class IMSnParsAnalysisTool(object):
    __metaclass__ = abc.ABCMeta
    
    
    @abc.abstractmethod
    def runAnalysis(self, sentences, output):
        pass
    
    
class SurfGradientAnalysis(IMSnParsAnalysisTool):
    
    def __init__(self, reprLstmBuilder, lstmLayer):
        self.__reprLstmBuilder = reprLstmBuilder
        self.__lstmLayer = lstmLayer
        
    def __addToResult(self, name, pos, tokId, grad, sentGrad, result):
        if name not in result:
            result[name] = { }
            
        if pos not in result[name]:
            result[name][pos] = [ ]
        
        result[name][pos].append(((tokId, grad, sentGrad)))
    
    def __dumpResult(self, sentId, result, out):
        outFormat = "[%s] sent=%i pos=%i grad=%s full=%s\n"
        
        toPrint = [ ]
        names = sorted(result.keys())
        for name in names:
            poss = sorted(result[name].keys())
            for pos in poss:
                values = result[name][pos]
                for tokId, grad, sentGrad in values:
                    toPrint.append((sentId, tokId, name, pos, str(grad), str(sentGrad)))
                        
        toPrint.sort()
        for sentId, tokId, name, pos, grad, sentGrad in toPrint:
            out.write(outFormat % (name, sentId, pos, grad, sentGrad))
                
    def runAnalysis(self, sentences, output):
        out = open(output, "w")
        
        for sentId, sent in enumerate(sentences):
            dynet.renew_cg()
            
            result = { }
            
            instance = self.__reprLstmBuilder.buildInstance(sent)
            embeds = self.__reprLstmBuilder.prepareEmbeddingsForLSTM(instance, isTraining=False)
            vectors = self.__reprLstmBuilder.applyLSTMToEmbeds(embeds, isTraining = False)
            
            correctTree = datatypes.sentence2Tree(sent)
            for pos in range(len(sent)):
                # prepare info about the tree
                children = sorted(correctTree.getChildren(pos))
                headPos = correctTree.getHead(pos)
                
                if headPos != -1:
                    grandPos = correctTree.getHead(headPos)
                else:
                    grandPos = None
                
                sibl = [ ]
                for child in correctTree.getChildren(headPos):
                    if child != pos:
                        sibl.append(child)
                
                
                # back-propagate -- all positions
                dynet.sum_elems(vectors.wordsV[pos][self.__lstmLayer - 1]).backward()
                wordGradients = [ _getGradient(embed) for embed in embeds.wordsV ]
                sentGrad = sum(wordGradients)
                
                # write for every position
                for cPos in range(len(sent)):
                    grad = wordGradients[cPos]
                    if cPos == headPos:
                        self.__addToResult("Head", pos - cPos, pos, grad, sentGrad, result)
                    elif cPos == grandPos:
                        self.__addToResult("Grand", pos - cPos, pos, grad, sentGrad, result)
                    elif cPos in children:
                        self.__addToResult("Child", pos - cPos, pos, grad, sentGrad, result)
                    elif cPos in sibl:
                        self.__addToResult("Sibl", pos - cPos, pos, grad, sentGrad, result)
                    else:
                        self.__addToResult("Other", pos - cPos, pos, grad, sentGrad, result)
    
            self.__dumpResult(sentId, result, out)
            
            

class FeatGradientAnalysis(IMSnParsAnalysisTool):
    
    def __init__(self, featAnalyzer, reprLstmBuilder):
        self.__featAnalyzer = featAnalyzer
        self.__reprLstmBuilder = reprLstmBuilder

    def runAnalysis(self, sentences, output):
        out = open(output, "w")
        outFormat = "[%s] fId=%i grad=%s full=%s\n"
        
        for sent in sentences:
            instance = self.__reprLstmBuilder.buildInstance(sent)
            
            dynet.renew_cg()
            self.__featAnalyzer.renewNetwork()
                
            embeds = self.__reprLstmBuilder.prepareEmbeddingsForLSTM(instance, isTraining=False)
            vectors = self.__reprLstmBuilder.applyLSTMToEmbeds(embeds, isTraining = False)
            
            grads = self.__featAnalyzer.analyzeFeatures(instance, embeds, vectors)
            for (fId, fName, fGrad, sumGrad) in grads:
                assert type(fName) == str
                out.write(outFormat % ( fName, fId, str(fGrad), str(sumGrad)))

class IMSnParsFeatAnalyzer(object):
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def analyzeFeatures(self, instance, beforeLstm, vectors):
        pass
    
    @abc.abstractmethod
    def renewNetwork(self):
        pass
    
    
    
class DummyFeatOutputCache(object):
    def __init__(self):
        pass

    def add(self, featId, feat, featOut):
        pass
        
    def contains(self, featId, feat):
        return False
    
class TransFeatAnalyzer(IMSnParsFeatAnalyzer):
    
    def __init__(self, transTask):
        self.__logger = logging.getLogger(self.__class__.__name__)
        self.__tsystem = transTask.getTransitionSystem()
        self.__featReprBuilder = transTask.getFeatReprBuilder()
        self.__network = transTask.getNetwork()
    
    def renewNetwork(self):
        self.__network.renewNetwork()
    
    def analyzeFeatures(self, instance, beforeLstm, vectors):
        result = [ ]
        
        cache = DummyFeatOutputCache()
        state = self.__tsystem.initialState(len(instance.sentence))
        cId = 0
        while not self.__tsystem.isFinal(state):
            featReprs = self.__featReprBuilder.extractAllAndBuildFeatRepr(state, cache, vectors, isTraining=False)
            netOut = self.__network.buildOutput(dynet.esum(featReprs), isTraining=False)
            netVal = netOut.value()
             
            scoredTransIds = np.argsort(netVal)[::-1]
            bestTransId = utils.first(lambda x : self.__tsystem.isValidTransition(state, x), scoredTransIds)
             
            if bestTransId == None:
                msg = "No valid transition in state: %s" % str(state)
                self.__logger.error(msg)
                raise RuntimeError(msg)
             
            predicted = bestTransId
            if isinstance(predicted, np.int64):
                predicted = predicted.item()
              
            score = netOut[predicted]
            score.backward()
            
            wordGradients = [ _getGradient(embed) for embed in beforeLstm.wordsV ]
            rootGradient = _getGradient(beforeLstm.rootV)
            
            gSum = sum(wordGradients) + rootGradient
            
            allFeatures = { }
            self.__collectFeatures(state.stack.toList()[::-1], "S", state, allFeatures)
            self.__collectFeatures(state.buffer.toList(), "B", state, allFeatures)
            
            for sentPos, featName in allFeatures.items():
                if sentPos == datatypes.Tree.ROOT:
                    result.append((cId, featName, rootGradient, gSum))
                else:
                    result.append((cId, featName, wordGradients[sentPos], gSum))
                            
            self.__tsystem.applyTransition(state, bestTransId)
            cId += 1
    
        return result
  
    def __collectFeatures(self, elems, name, state, feats):
        for fPos, sentPos in enumerate(elems):
            fName = name + str(fPos)
            
            assert sentPos not in feats
            feats[sentPos] = fName
            
            elC = sorted(state.arcs.getChildren(sentPos))
            
            # left children
            for cPos, cSentPos in enumerate(elC):
                if cSentPos > sentPos:
                    break
                
                cFeatName = "lm" + "+" + str(cPos)
                cName = fName + cFeatName
                assert cSentPos not in feats
                feats[cSentPos] = cName
            
            # right children
            elC.sort(reverse=True)
            for cPos, cSentPos in enumerate(elC):
                if cSentPos < sentPos:
                    break
                
                cFeatName = "rm" + "-" + str(cPos)
                cName = fName + cFeatName
                
                assert cSentPos not in feats
                feats[cSentPos] = cName
            
    
class GraphFeatAnalyzer(IMSnParsFeatAnalyzer):
    
    def __init__(self, graphTask):
        self.__logger = logging.getLogger(self.__class__.__name__)
        self.__mst = graphTask.getMSTAlg()
        self.__decoder = graphTask.getDecoder()
        self.__featReprBuilder = graphTask.getFeatReprBuilder()
        self.__network = graphTask.getNetwork()
    
    def renewNetwork(self):
        self.__network.renewNetwork()
    
    def analyzeFeatures(self, instance, beforeLstm, vectors):
        scores = self.__mst.emptyScores(instance)
        self.__decoder.calculateScores(instance, vectors, self.__network, scores, isTraining=False)
        predictTree = self.__mst.findMSTTree(scores)
         
        result = [ ]
        for dId in range(len(instance.sentence)):
            hId = predictTree.getHead(dId)
             
            score = scores.getOutput(hId, dId)
            score.backward()
             
            features = self.__collectFeatures(hId, dId, predictTree)
             
            wordGradients = [ _getGradient(embed) for embed in beforeLstm.wordsV ]
            rootGradient = _getGradient(beforeLstm.rootV)
            gSum = sum(wordGradients) + rootGradient
 
            for (sentPos, featName) in features:
                if sentPos == datatypes.Tree.ROOT:
                    result.append((dId, featName, rootGradient, gSum))
                else:
                    result.append((dId, featName, wordGradients[sentPos], gSum))
                 
        return result
     
    def __collectFeatures(self, hId, dId, predictTree):
        result = [ (hId, "H"), ( dId, "D") ]
         
        sibl = [ ]
        for child in predictTree.getChildren(hId):
            if child != dId:
                sibl.append(child)
         
        for s in sibl:
            result.append((s, "S"))
             
        children = sorted(predictTree.getChildren(dId))    
        for c in children:
            result.append((c, "C"))
 
        if hId != -1:
            grandPos = predictTree.getHead(hId)
            result.append((grandPos, "G"))
        else:
            grandPos = None
         
        for i in range(predictTree.nrOfTokens()):
            if i in sibl or i in children or i == hId or i == dId or i == grandPos:
                continue
                 
            result.append((i, "H" + str(hId - i)))
            result.append((i, "D" + str(dId - i)))
                 
        return result
