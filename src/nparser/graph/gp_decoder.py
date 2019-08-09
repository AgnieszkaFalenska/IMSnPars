'''
Created on 23.08.2017

@author: falensaa
'''

import abc
import dynet

from nparser.graph.gp_features import FeatId

class GraphDecoder(abc.ABC):
    
    @abc.abstractmethod
    def findErrors(self, outputs, correct, predicted):
        pass
    
    @abc.abstractmethod
    def calculateScores(self, instance, outBuilder, network, scores, isTraining):
        pass

    @abc.abstractmethod
    def augmentScores(self, scores, instance, cost = 1.0):
        pass

class FirstOrderDecoder(GraphDecoder):
    def __init__(self, featReprBuilder):
        self.__featReprBuilder = featReprBuilder
        
    def findErrors(self, outputs, correct, predicted):
        errs = [ tPos for tPos in range(correct.nrOfTokens()) if correct.getHead(tPos) != predicted.getHead(tPos) ]
           
        if len(errs) == 0:
            return [ ]
           
        return [ (outputs.getOutput(predicted.getHead(tPos), tPos), outputs.getOutput(correct.getHead(tPos), tPos)) for tPos in errs ]

    #@profile
    def calculateScores(self, instance, vectors, network, scores, isTraining):
        dReprCache = { }
        for dId in range(-1, len(instance.sentence)):
            depReprs = self.__featReprBuilder.extractAndBuildFeatRepr(FeatId.DEP, dId, instance.sentence, vectors, isTraining)
            depRepr = dynet.esum(depReprs) if len(depReprs) > 0 else None
            dReprCache[dId] = (depRepr, len(depReprs))
        
        for hId in range(-1, len(instance.sentence)):
            headReprs = self.__featReprBuilder.extractAndBuildFeatRepr(FeatId.HEAD, hId, instance.sentence, vectors, isTraining)
            headRepr = dynet.esum(headReprs) if len(headReprs) > 0 else None
            
            for dId in range(-1, len(instance.sentence)):
                depRepr, depNr = dReprCache[dId]
                distRepr = self.__featReprBuilder.onlyBuildFeatRepr(FeatId.DIST, (hId, dId), isTraining)
                
                featRepr = [ headRepr, depRepr ]
                featReprNr = len(headReprs) + depNr
                if distRepr != None:
                    featRepr.append(distRepr)
                    featReprNr += 1
                
                assert featReprNr == self.__featReprBuilder.getNrOfFeatures()
                featRepr = dynet.esum([ f for f in featRepr if f is not None])
                netOut = network.buildOutput(featRepr, isTraining=isTraining)
                
                scores.addOutput(hId, dId, netOut)
                scores.addScore(hId, dId, dynet.max_dim(netOut).scalar_value())
    
    def augmentScores(self, scores, instance, cost = 1.0):
        for hId in range(-1, len(instance.sentence)):
            for dId in range(0, len(instance.sentence)):
                if hId != instance.sentence[dId].getHeadPos():
                    oldScore = scores.getScore(hId, dId)
                    scores.addScore(hId, dId, oldScore + cost)
   