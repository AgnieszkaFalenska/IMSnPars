'''
Created on 23.08.2017

@author: falensaa
'''

from enum import IntEnum

class FeatId(IntEnum):
    S0 = 1
    S1 = 2
    S2 = 3
    B0 = 4
    B1 = 5
    B2 = 6
    S0LMC = 7
    S0RMC = 8
    S1LMC = 9
    S1RMC = 10
    S2LMC = 11
    S2RMC = 12
    B0LMC = 13
    B0RMC = 14
    S0LMC_2 = 15
    S0RMC_2 = 16
    S0LMC_LMC = 17
    S0RMC_RMC = 18
    S1LMC_LMC = 19
    S1RMC_RMC = 20
    
class StateFeatExtractor(object):
    def __init__(self, featId):
        self.__featId = featId
        
        allFeats = {FeatId.S0 : self.__s0Feat,
                    FeatId.S1 : self.__s1Feat,
                    FeatId.S2 : self.__s2Feat,
                    FeatId.B0 : self.__b0Feat,
                    FeatId.B1 : self.__b1Feat,
                    FeatId.B2 : self.__b2Feat,
                    FeatId.S0LMC : self.__s0lmc,
                    FeatId.S0RMC : self.__s0rmc,
                    FeatId.S1LMC : self.__s1lmc,
                    FeatId.S1RMC : self.__s1rmc,
                    FeatId.S2LMC : self.__s2lmc,
                    FeatId.S2RMC : self.__s2rmc,
                    FeatId.B0LMC : self.__b0lmc,
                    FeatId.B0RMC : self.__b0rmc,
                    FeatId.S0LMC_2 : self.__s0lmc_2,
                    FeatId.S0RMC_2 : self.__s0rmc_2,
                    FeatId.S0LMC_LMC : self.__s0lmc_lmc,
                    FeatId.S0RMC_RMC : self.__s0rmc_rmc,
                    FeatId.S1LMC_LMC : self.__s1lmc_lmc,
                    FeatId.S1RMC_RMC : self.__s1rmc_rmc }
        
        self.__featFun = allFeats[self.__featId]
            
    def extractFeatures(self, state):
        return self.__featFun(state)
    
    def getFeatId(self):
        return self.__featId
    
    def __s0Feat(self, state):
        return state.getStackElem(0)
    
    def __s1Feat(self, state):
        return state.getStackElem(1)
    
    def __s2Feat(self, state):
        return state.getStackElem(2)
    
    def __b0Feat(self, state):
        return state.getBufferElem(0)
    
    def __b1Feat(self, state):
        return state.getBufferElem(1)
    
    def __b2Feat(self, state):
        return state.getBufferElem(2)
    
    def __s0lmc(self, state):
        s0 = self.__s0Feat(state)
        s0lmc_barch = state.getLeftMostChild(s0)
        if s0lmc_barch is None:
            return s0
        
        return s0lmc_barch
        
    def __s0rmc(self, state):
        s0 = self.__s0Feat(state)
        s0rmc_barch = state.getRightMostChild(s0)
        if s0rmc_barch is None:
            return s0
        
        return s0rmc_barch
    
    def __s1lmc(self, state):
        s1 = self.__s1Feat(state)
        s1lmc = state.getLeftMostChild(s1)
        if s1lmc is None:
            return s1
        
        return s1lmc
        
    def __s1rmc(self, state):
        s1 = self.__s1Feat(state)
        s1rmc = state.getRightMostChild(s1)
        if s1rmc is None:
            return s1
        
        return s1rmc
    
    def __s2lmc(self, state):
        s2 = self.__s2Feat(state)
        s2lmc = state.getLeftMostChild(s2)
        if s2lmc is None:
            return s2
        
        return s2lmc
        
    def __s2rmc(self, state):
        s2 = self.__s2Feat(state)
        s2rmc = state.getRightMostChild(s2)
        if s2rmc is None:
            return s2
        
        return s2rmc
    
    def __b0lmc(self, state):
        b0 = self.__b0Feat(state)
        b0lmc = state.getLeftMostChild(b0)
        if b0lmc is None:
            return b0
        
        return b0lmc
        
    def __b0rmc(self, state):
        b0 = self.__b0Feat(state)
        b0rmc = state.getRightMostChild(b0)
        if b0rmc is None:
            return b0
        
        return b0rmc
        
    def __s0lmc_2(self, state):
        s0lmc_2 = state.getLeftSecondChild(self.__s0Feat(state))
        return s0lmc_2
    
    def __s0rmc_2(self, state):
        s0rmc_2 = state.getRightSecondChild(self.__s0Feat(state))
        return s0rmc_2
    
    def __s0lmc_lmc(self, state):
        s0lmc_lmc = state.getLeftMostChild(self.__s0lmc(state))
        return s0lmc_lmc
    
    def __s0rmc_rmc(self, state):
        s0rmc_rmc = state.getRightMostChild(self.__s0rmc(state))
        return s0rmc_rmc

    def __s1lmc_lmc(self, state):
        s1lmc_lmc = state.getLeftMostChild(self.__s1lmc(state))
        return s1lmc_lmc
    
    def __s1rmc_rmc(self, state):
        s1rmc_rmc = state.getRightMostChild(self.__s1rmc(state))
        return s1rmc_rmc
    
    def __str__(self):
        return str(self.__featId)
        
    
class TransFeatureExtractor(object):
    
    def __init__(self, stateExtractors):
        assert type(stateExtractors) == type([])
        self.__stateExtractors = stateExtractors
    
    def getNrOfFeaturs(self):
        return len(self.__stateExtractors)
    
    def getFeatIds(self):
        return [ feat.getFeatId() for feat in self.__stateExtractors ]
    
    def extractAllFeatures(self, state):
        return [ ( feat.getFeatId(), feat.extractFeatures(state)) for feat in self.__stateExtractors ]
    
    