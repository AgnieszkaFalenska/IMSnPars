'''
Created on 23.08.2017

@author: falensaa
'''

import abc
import dynet

from nparser.graph import features as gfeatures

class GraphDecoder(object):
    __metaclass__ = abc.ABCMeta
    
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
        
    def getFeatReprBuilder(self):
        return self.__featReprBuilder
    
    def findErrors(self, outputs, correct, predicted):
        errs = [ tPos for tPos in range(correct.nrOfTokens()) if correct.getHead(tPos) != predicted.getHead(tPos) ]
           
        if len(errs) == 0:
            return [ ]
           
        return [ (outputs.getOutput(predicted.getHead(tPos), tPos), outputs.getOutput(correct.getHead(tPos), tPos)) for tPos in errs ]

    #@profile
    def calculateScores(self, instance, vectors, network, scores, isTraining):
        dReprCache = { }
        for dId in range(-1, len(instance.sentence)):
            depReprs = self.__featReprBuilder.extractAndBuildFeatRepr(gfeatures.FeatId.DEP, dId, instance.sentence, vectors, isTraining)
            depRepr = dynet.esum(depReprs) if len(depReprs) > 0 else None
            dReprCache[dId] = (depRepr, len(depReprs))
        
        for hId in range(-1, len(instance.sentence)):
            headReprs = self.__featReprBuilder.extractAndBuildFeatRepr(gfeatures.FeatId.HEAD, hId, instance.sentence, vectors, isTraining)
            headRepr = dynet.esum(headReprs) if len(headReprs) > 0 else None
            
            for dId in range(-1, len(instance.sentence)):
                depRepr, depNr = dReprCache[dId]
                distRepr = self.__featReprBuilder.onlyBuildFeatRepr(gfeatures.FeatId.DIST, (hId, dId), isTraining)
                
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
                    
                    
class SecondOrderGrandparentDecoder(GraphDecoder):
    def __init__(self, featReprBuilder):
        self.__featReprBuilder = featReprBuilder
        
    def findErrors(self, outputs, correct, predicted):
        corrArcs = self.__getArcs(correct)
        predArcs = self.__getArcs(predicted)
        
        assert len(corrArcs) == correct.nrOfTokens()
        assert len(predArcs) == correct.nrOfTokens()
        
        errors = [ ]
        for tokId in range(correct.nrOfTokens()):
            if corrArcs[tokId] == predArcs[tokId]:
                continue
            
            corrGr, corrH = corrArcs[tokId]
            predGr, predH = predArcs[tokId]
            
            errors.append((outputs.getOutput(predGr, predH, tokId), outputs.getOutput(corrGr, corrH, tokId)))
            
        return errors
    
    def calculateScores(self, instance, vectors, network, scores, isTraining):
        if isTraining:
            corrArcs = self.__getArcs(instance.correctTree)
        else:
            corrArcs = None
            
        gReprCache = { }
        for gId in range(-1, len(instance.sentence)):
            grandReprs = self.__featReprBuilder.extractAndBuildFeatRepr(gfeatures.FeatId.GRAND, gId, instance.sentence, vectors, isTraining)
            grandRepr = dynet.esum(grandReprs) if len(grandReprs) > 0 else None
            gReprCache[gId] = ( grandRepr, len(grandReprs))
            
        hReprCache = { }
        for hId in range(-1, len(instance.sentence)):
            headReprs = self.__featReprBuilder.extractAndBuildFeatRepr(gfeatures.FeatId.HEAD, hId, instance.sentence, vectors, isTraining)
            headRepr = dynet.esum(headReprs) if len(headReprs) > 0 else None
            hReprCache[hId] = ( headRepr, len(headReprs))
            
        dReprCache = { }
        for dId in range(-1, len(instance.sentence)):
            depReprs = self.__featReprBuilder.extractAndBuildFeatRepr(gfeatures.FeatId.DEP, dId, instance.sentence, vectors, isTraining)
            depRepr = dynet.esum(depReprs) if len(depReprs) > 0 else None
            dReprCache[dId] = ( depRepr, len(depReprs))
        
        distReprCache = { }
        toFill = self.__scoresToFill(instance, corrArcs)
        for (gId, hId, dId) in toFill:
            if (hId, dId) in distReprCache:
                distRepr = distReprCache[(hId, dId)]
            else:
                distRepr = self.__featReprBuilder.onlyBuildFeatRepr(gfeatures.FeatId.DIST, (hId, dId), isTraining)
                distReprCache[(hId, dId)] = distRepr
            
            dRepr, dNr = dReprCache[dId]
            hRepr, hNr = hReprCache[hId]
            gRepr, gNr = gReprCache[gId]
                
            featRepr = [ dRepr, hRepr, gRepr ]
            featReprNr = dNr + hNr + gNr
            
            if distRepr != None:
                featRepr.append(distRepr)
                featReprNr += 1
                    
            assert featReprNr == self.__featReprBuilder.getNrOfFeatures()
            featRepr = dynet.esum([ f for f in featRepr if f is not None])
            netOut = network.buildOutput(featRepr, isTraining=isTraining)     
            
            scores.addOutput(gId, hId, dId, netOut)
            scores.addScore(gId, hId, dId, netOut.scalar_value())
    
    def augmentScores(self, scores, instance, cost = 1.0):
        corrArcs = self.__getArcs(instance.correctTree)
        toFill = self.__scoresToFill(instance, corrArcs)
        for (gId, hId, dId) in toFill:
            if dId in corrArcs and corrArcs[dId] != (gId, hId):
                oldScore = scores.getScore(gId, hId, dId)
                scores.addScore(gId, hId, dId, oldScore + cost)
                
    def __scoresToFill(self, instance, corrArcs):
        scores = { }
        length = len(instance.sentence) + 1
        
        #2.loop
        for k in range(1, length):
            # the distance k
            for s in range(length):
                # span between s and t
                t = s + k
                if t >= length:
                    break
                
                for g in range(length):
                    # grand-node g
                    if g >= s and g <= t and g!=0: # special node when g==0 (0,0,m)
                        continue
                    
                    scores[(g-1,t-1,s-1)] = 1
                    scores[(g-1,s-1,t-1)] = 1
               
        if corrArcs: 
            for (d, (g, h)) in corrArcs.items():
                scores[(g,h,d)] = 1
            
        return scores
    
    def __getArcs(self, tree):
        result = { }
        
        for tokId in range(tree.nrOfTokens()):
            headId = tree.getHead(tokId)
            if headId != -1:
                grandId = tree.getHead(headId)
            else:
                grandId = headId
                
            result[tokId] = (grandId, headId)
            
        return result
 
class SecondOrderSiblDecoder(GraphDecoder):
    
    def __init__(self, featReprBuilder):
        self.__featReprBuilder = featReprBuilder
        
    def findErrors(self, scores, correct, predicted):
        corrArcs = self.__getArcs(correct)
        predArcs = self.__getArcs(predicted)
         
        assert len(corrArcs) == correct.nrOfTokens()
        assert len(predArcs) == correct.nrOfTokens()
         
        errors = [ ]
        for tokId in range(correct.nrOfTokens()):
            if corrArcs[tokId] == predArcs[tokId]:
                continue
             
             
            corrH, corrC = corrArcs[tokId]
            predH, predC = predArcs[tokId]
             
            errors.append((scores.getOutput(predH, predC, tokId), scores.getOutput(corrH, corrC, tokId)))
             
        return errors
     
    def calculateScores(self, instance, vectors, network, scores, isTraining):
        hReprCache = { }
        for hId in range(-1, len(instance.sentence)):
            headReprs = self.__featReprBuilder.extractAndBuildFeatRepr(gfeatures.FeatId.HEAD, hId, instance.sentence, vectors, isTraining)
            headRepr = dynet.esum(headReprs) if len(headReprs) > 0 else None
            hReprCache[hId] = ( headRepr, len(headReprs))
            
        sReprCache = { }
        for sId in range(-1, len(instance.sentence)):
            siblReprs = self.__featReprBuilder.extractAndBuildFeatRepr(gfeatures.FeatId.SIBL, sId, instance.sentence, vectors, isTraining)
            siblRepr = dynet.esum(siblReprs) if len(siblReprs) > 0 else None
            sReprCache[sId] = ( siblRepr, len(siblReprs))
            
        dReprCache = { }
        for dId in range(-1, len(instance.sentence)):
            depReprs = self.__featReprBuilder.extractAndBuildFeatRepr(gfeatures.FeatId.DEP, dId, instance.sentence, vectors, isTraining)
            depRepr = dynet.esum(depReprs) if len(depReprs) > 0 else None
            dReprCache[dId] = ( depRepr, len(depReprs))
        
        distReprCache = { }
        toFill = self.__scoresToFill(instance)
        for (hId, cId, dId) in toFill:
            if (hId, dId) in distReprCache:
                distRepr = distReprCache[(hId, dId)]
            else:
                distRepr = self.__featReprBuilder.onlyBuildFeatRepr(gfeatures.FeatId.DIST, (hId, dId), isTraining)
                distReprCache[(hId, dId)] = distRepr
            
            dRepr, dNr = dReprCache[dId]
            hRepr, hNr = hReprCache[hId]
            cRepr, cNr = sReprCache[cId]
                
            featRepr = [ dRepr, hRepr, cRepr ]
            featReprNr = dNr + hNr + cNr
            if distRepr != None:
                featRepr.append(distRepr)
                featReprNr += 1
                    
            assert featReprNr == self.__featReprBuilder.getNrOfFeatures()
            featRepr = dynet.esum([ f for f in featRepr if f is not None])
            netOut = network.buildOutput(featRepr, isTraining=isTraining)     
            
            scores.addOutput(hId, cId, dId, netOut)
            scores.addScore(hId, cId, dId, netOut.scalar_value())
    
    def augmentScores(self, scores, instance, cost = 1.0):
        corrArcs = self.__getArcs(instance.correctTree)
        toFill = self.__scoresToFill(instance)
        for (hId, cId, dId) in toFill:
            if dId in corrArcs and corrArcs[dId] != (hId, cId):
                oldScore = scores.getScore(hId, cId, dId)
                scores.addScore(hId, cId, dId, oldScore + cost)
            
                
    def __scoresToFill(self, instance):
        length = len(instance.sentence) + 1
        
        result = { }
        #the distance k
        for dist in range(1, length):
            # span between s and t
            for s in range(length):
                t = s + dist
                if( t >= length):
                    break;
                
                result[(t-1,t-1,s-1)] = 1
                
                # others
                for r in range(s+1, t):
                    result[(t-1,r-1,s-1)] = 1
                    
                result[(s-1,s-1,t-1)] = 1
                
                #others
                for r in range(s+1, t):
                    result[(s-1,r-1,t-1)] = 1
                    
        return result
    
    def __getArcs(self, tree):
        result = { }
        
        children = { }
        for tokId in range(tree.nrOfTokens()):
            headId = tree.getHead(tokId)
            
            if headId not in children:
                children[headId] = { }
                
            children[headId][tokId] = 1
            
        for headId in children:
            cIds = list(children[headId].keys())
            cIds.sort()
            
            before = [ ]
            after = [ ]
            for cId in cIds:
                if cId < headId:
                    before.append(cId)
                else:
                    after.append(cId)
            
            before.sort(reverse=True)
            if len(before) > 0:
                result[before[0]] = (headId, headId)
                
            if len(before) > 1:
                for i in range(0, len(before) - 1):
                    result[before[i+1]] = (headId, before[i])
            
            if len(after) > 0:
                result[after[0]] = (headId, headId)
                
            if len(after) > 1:
                for i in range(0, len(after) - 1):
                    result[after[i+1]] = (headId, after[i])        
                
        return result
