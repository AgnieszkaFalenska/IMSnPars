'''
Created on 23.08.2017

@author: falensaa
'''

import abc, random

class Oracle(object):
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def handlesNonProjectiveTrees(self):
        return
    
    @abc.abstractmethod
    def handlesExploration(self):
        return
    
    @abc.abstractmethod
    def nextCorrectTransitions(self, state, tree):
        return
    
class ExplorePolicy(object):
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def doCorrect(self, correctVal, predictedVal, epoch = 1):
        return
    
class OriginalExplorePolicy(ExplorePolicy):
    def __init__(self, firstCorrect, predProp):
        self.__firstCorrect = firstCorrect
        self.__predProp = predProp
        
    def doCorrect(self, correctVal, predictedVal, epoch = 1):
        return epoch < self.__firstCorrect or random.random() > self.__predProp
         
                     
class AggresiveExplorePolicy(ExplorePolicy):
    def __init__(self, pagg):
        self.__pagg = pagg
        
    def doCorrect(self, correctVal, predictedVal, epoch = 1):
        return correctVal > predictedVal and random.random() > self.__pagg


def buildStaticCorrectTransitions(tree, tsystem, oracle, labeler = None):
    if labeler == None:
        transSystem = tsystem
    else:
        transSystem = labeler
        
    result = [ ]
    state = transSystem.initialState(tree.nrOfTokens())

    while not transSystem.isFinal(state):
        transition = oracle.nextCorrectTransitions(state, tree)[0]
        
        if transition == None:
            break
        
        result.append(transition)
        transSystem.applyTransition(state, transition)
    
    return result

                    