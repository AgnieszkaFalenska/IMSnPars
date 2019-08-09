'''
Created on 18.08.2017

@author: falensaa
'''

import logging

from nparser.trans.tsystem import Transition, TransSystem, State
from nparser.trans import oracle

class LeftArc(Transition):
    
    def _buildArc(self, state):
        stackTop = state.stack.top()
        stackSecond = state.getStackElem(1)
        return (stackTop, stackSecond)
        
    def update(self, state):
        stackTop = state.stack.pop()
        stackSecond = state.stack.pop()
        state.stack.push(stackTop)
        state.arcs.add(stackTop, stackSecond)
    
    def isValid(self, state):
        if state.stack.length() < 2:
            return False
        
        stackSecond = state.getStackElem(1)
        return stackSecond != State.ROOT
    
class RightArc(Transition):
    
    def _buildArc(self, state):
        stackTop = state.stack.top()
        stackSecond = state.getStackElem(1)
        return (stackSecond, stackTop)
    
    def update(self, state):
        stackTop = state.stack.pop()
        stackSecond = state.stack.top()
        state.arcs.add(stackSecond, stackTop)

    def isValid(self, state):
        if state.stack.length() < 2:
            return False
        
        return state.stack.top() != State.ROOT
    
class Shift(Transition):
    
    def _buildArc(self, _):
        return None
    
    def update(self, state):
        frontBuffer = state.buffer.pop()
        state.stack.push(frontBuffer)
  
    def isCorrect(self, state, _):
        return self.isValid(state)

    def isValid(self, state):
        return not state.buffer.empty() 
        
class ArcStandard(TransSystem):
    
    SHIFT = 0
    LEFTARC = 1
    RIGHTARC = 2
       
    __SHIFT = Shift()
    __LEFTARC = LeftArc()
    __RIGHTARC = RightArc()
    
    def applyTransition(self, state, tId):
        self.__getTransition(tId).update(state)
    
    def isValidTransition(self, state, tId):
        return self.__getTransition(tId).isValid(state)

    def isFinal(self, state):
        return state.buffer.empty() and state.stack.length() == 1
    
    def initialState(self, nrOfTokens):
        return State(nrOfTokens, True)
    
    def getNrOfTransitions(self):
        return 3
    
    # functions for labeler
    def _buildArc(self, tId, state):
        return self.__getTransition(tId)._buildArc(state)
        
    def _getNonArcTransitions(self):
        return [ self.SHIFT ] 
    
    def _getArcTransitions(self):
        return [ self.LEFTARC, self.RIGHTARC ]
    
    # private functions
    
    def __getTransition(self, tId):
        if tId == self.SHIFT:
            return self.__SHIFT
        elif tId == self.LEFTARC:
            return self.__LEFTARC
        elif tId == self.RIGHTARC:
            return self.__RIGHTARC
        else:
            raise RuntimeError("Unknown transition id: %i" % tId)
    
    
class ArcStandardStaticOracle(oracle.Oracle):
    """Static oracle for ArcHybrid and ArcStandard"""
    
    def __init__(self, system, labeler = None):
        self.__logger = logging.getLogger(self.__class__.__name__)
        self.__system = system
        self.__labeler = labeler
        
    def handlesNonProjectiveTrees(self):
        return False
    
    def handlesExploration(self):
        return False
    
    def nextCorrectTransitions(self, state, tree):
        transId = self.__nextCorrectId(state, tree)
        if transId == None:
            return [ None ]
        
        return [ self._getLabeledTransition(transId, state, tree) ]

    def __nextCorrectId(self, state, tree):
        if state.stack.empty():
            return self.__system.SHIFT
        
        if self.__canLA(state, tree):
            return self.__system.LEFTARC
        elif self.__canRA(state, tree):
            return self.__system.RIGHTARC
        elif self.__canSH(state, tree):
            return self.__system.SHIFT
        
        return None
    
    def __canLA(self, state, tree):
        if not self.__system.isValidTransition(state, self.__system.LEFTARC):
            return False
        
        head, dep = self.__system._buildArc(self.__system.LEFTARC, state)
        return tree.hasArc(head, dep) and state.arcs.theSameChilden(dep, tree)
    
    def __canRA(self, state, tree):
        if not self.__system.isValidTransition(state, self.__system.RIGHTARC):
            return False
         
        head, dep = self.__system._buildArc(self.__system.RIGHTARC, state)
        return tree.hasArc(head, dep) and state.arcs.theSameChilden(dep, tree)

    def __canSH(self, state, tree):
        return self.__system.isValidTransition(state, self.__system.SHIFT)
    
    def _getLabeledTransition(self, tId, state, tree):
        if self.__labeler == None:
            return tId 
        else:
            if tId == self.__system.SHIFT:
                return self.__labeler.getLblTrans(self.__system.SHIFT)
            else:
                tArc = self.__system._buildArc(tId, state)
                lbl = tree.getLabel(tArc[1])
                return self.__labeler.getLblTrans(tId, lbl)
        