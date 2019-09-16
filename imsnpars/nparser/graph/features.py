'''
Created on 23.08.2017

@author: falensaa
'''

from enum import IntEnum

class FeatId(IntEnum):
    HEAD = 1
    DEP = 2
    HEAD_P_1 = 3
    HEAD_P_2 = 4
    HEAD_M_1 = 5
    HEAD_M_2 = 6
    DEP_P_1 = 7
    DEP_P_2 = 8
    DEP_M_1 = 9
    DEP_M_2 = 10
    SIBL = 11
    GRAND = 12
    DIST = 13
    
class TokenFeatExtractor(object):
    def __init__(self):
        self.__featIds = [ ]
        self.__shifts = [ ]
        
    def getFeatIds(self):
        return self.__featIds
    
    def extractFeatures(self, feat, sentence):
        result = [ ]
        for featId, shift in zip(self.__featIds, self.__shifts):
            sentPos = feat + shift
            if sentPos >= -1 and sentPos < len(sentence):
                result.append((featId, sentPos))
            else:
                result.append((featId, None))
                
        return result 
    
    def addShift(self, featId, shift):
        self.__featIds.append(featId)
        self.__shifts.append(shift)
        
    def __str__(self):
        return ",".join([ self.__shiftToStr(fI, sh) for (fI, sh) in zip(self.__featIds, self.__shifts)])
    
    def __shiftToStr(self, featId, shift):
        if shift > 0:
            shStr = "+" + str(shift)
        elif shift == 0:
            shStr = ""
        else:
            shStr = str(shift)
            
        return str(featId) + shStr
        
#TODO:  change the dimension of the dist feature
class DistFeatureBuilder(object):
    def __init__(self, maxDist = 7):
        self.__featId = FeatId.DIST
        self.__maxDist = maxDist
        self.__lookup = None
         
    def getFeatId(self):
        return self.__featId
    
    def buildRepresentation(self, feat, isTraining):
        hId, dId = feat
        dist = abs(hId - dId)
        if dist > self.__maxDist:
            dist = self.__maxDist
             
        return self.__lookup[dist]
         
    def initializeParameters(self, model, inputDim):
        #TODO:  fill only part of the vector
        self.__lookup = model.add_lookup_parameters((self.__maxDist + 1, inputDim))

    def __str__(self):
        return str(self.__featId) + ":" + str(self.__maxDist)
    
class GraphFeatureExtractor(object):
    
    def __init__(self, tokExtractors):
        assert type(tokExtractors) == type({})
        self.__tokExtractors = tokExtractors

    def extractFeatures(self, featId, sentence, feat):
        if featId in self.__tokExtractors:
            return self.__tokExtractors[featId].extractFeatures(feat, sentence)
        
        return [ ]
    
    def getFeatIds(self):
        return sum([ feat.getFeatIds() for feat in self.__tokExtractors.values()], [])
    
    
