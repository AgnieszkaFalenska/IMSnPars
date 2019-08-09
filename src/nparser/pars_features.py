'''
Created on Jun 29, 2018

@author: falensaa
'''

import dynet
import logging

class FeatOutputCache(object):
    def __init__(self):
        self.__outs = { }

    def add(self, featId, feat, featOut):
        if featId not in self.__outs:
            self.__outs[featId] = { }
            
        featOuts = self.__outs[featId]
        featOuts[feat] = featOut
        
    def contains(self, featId, feat):
        return featId in self.__outs and feat in self.__outs[featId]
    
    def get(self, featId, feat):
        return self.__outs[featId][feat]


class FeatReprBuilder(object):
    def __init__(self, featExtractor, featBuilders, dummyVecBuilder, network, reprLayer):
        self.__featExtractor = featExtractor
        self.__featBuilders = featBuilders
        self.__dummyVecBuilder = dummyVecBuilder
        self.__network = network
        self.__reprLayer = reprLayer
        
        assert type(featBuilders) == type({})
        
        featIds = self.__featExtractor.getFeatIds() + [ feat.getFeatId() for feat in featBuilders.values() ]
        self.__dummyVecBuilder.setFeatIds(featIds)
        self.__nrOfFeatures = len(featIds)
    
    def getNrOfFeatures(self):
        return self.__nrOfFeatures
            
    def initializeParameters(self, model, inputDim):
        for feat in self.__featBuilders.values():
            feat.initializeParameters(model, inputDim)

        self.__dummyVecBuilder.initializeParameters(model)

    def __getVectorForId(self, featId, wId, vectors):
        if self.__reprLayer == None:
            layer = None
        else:
            layer = self.__reprLayer
        
        if wId == None:
            return self.__dummyVecBuilder.prepareVector(featId)
        elif wId == -1:
            return self.__getLayer(layer, vectors.rootV)
        else:
            return self.__getLayer(layer, vectors.wordsV[wId])
    
    def __getLayer(self, reprLayer, vector):
        if reprLayer == None:
            return vector
        else:
            return vector[self.__reprLayer]        

    def onlyBuildFeatRepr(self, featId, feat, isTraining):
        if featId not in self.__featBuilders:
            return None
        
        featVec = self.__featBuilders[featId].buildRepresentation(feat, isTraining)
        featRepr = self.__network.buildFeatOutput(featId, featVec, isTraining)
        return featRepr
    
    def extractAndBuildFeatRepr(self, featId, feat, data, vectors, isTraining):
        feats = self.__featExtractor.extractFeatures(featId, data, feat)
        result = [ ]
        for (fId, feat) in feats:
            featVec = self.__getVectorForId(fId, feat, vectors)
            featRepr = self.__network.buildFeatOutput(fId, featVec, isTraining)
            result.append(featRepr)
        
        return result
            
    def extractAllAndBuildFeatRepr(self, data, cache, vectors, isTraining):
        feats = self.__featExtractor.extractAllFeatures(data)
        result = [ ]
        for (featId, feat) in feats:
            if cache.contains(featId, feat):
                result.append(cache.get(featId, feat))
            else:
                featVec = self.__getVectorForId(featId, feat, vectors)
                featRepr = self.__network.buildFeatOutput(featId, featVec, isTraining)
                cache.add(featId, feat, featRepr)
                result.append(featRepr)
            
        assert len(result) == self.__nrOfFeatures
        return result
    
################################################################
# dummy
################################################################

class RandomDummyVectorBuilder(object):
    def __init__(self, inDim, outDim, nonLinFun, separate):
        self.__logger =  logging.getLogger(self.__class__.__name__)
        
        self.__inDim = inDim
        self.__outDim = outDim
        
        self.__separate = separate
        self.__featIds = [ ]
        
        self.__dummyVec = None
        self.__word2repr = None
        self.__word2reprBias = None
            
        self.__nonLinFun = nonLinFun
        
    def initializeParameters(self, model):
        # separate vectors for every feature
        if self.__separate:
            self.__dummyVec = { }
            self.__word2repr = { }
            self.__word2reprBias = { }
        
            for featId in self.__featIds:
                self.__word2repr[featId] = model.add_parameters((self.__outDim, self.__inDim))
                self.__word2reprBias[featId] = model.add_parameters((self.__outDim))
                self.__dummyVec[featId] = model.add_parameters((self.__inDim))
        else:
            self.__word2repr = model.add_parameters((self.__outDim, self.__inDim))
            self.__word2reprBias = model.add_parameters((self.__outDim))
            self.__dummyVec = model.add_parameters((self.__inDim))
    
    def prepareVector(self, featId):
        if self.__separate:
            return self.__nonLinFun(self.__word2repr[featId] *  self.__dummyVec[featId].expr() + self.__word2reprBias[featId].expr())
        else:
            return self.__nonLinFun(self.__word2repr.expr() *  self.__dummyVec.expr() + self.__word2reprBias.expr())
        
    def setFeatIds(self, featIds):
        if self.__separate:
            self.__featIds += featIds
            self.__logger.debug("Adding separate features: %s" % ",".join([str(fId) for fId in featIds])) 
        
class ZeroDummyVectorBuilder(object):
    def __init__(self, outDim):
        self.__outDim = outDim
        
    def initializeParameters(self, model):
        pass
    
    def prepareVector(self, featId):
        return dynet.zeros(self.__outDim)

    def setFeatIds(self, featIds):
        pass      
