'''
Created on 18.08.2017

@author: falensaa
'''

import logging

from nparser.trans.tsystem import tdatatypes
from nparser.trans.tsystem import oracle

class LeftArc(tdatatypes.Transition):
    
    def _buildArc(self, state):
        stackTop = state.stack.top()
        bufferFront = state.buffer.head()
        return (bufferFront, stackTop)
        
    def update(self, state):
        stackTop = state.stack.pop()
        bufferFront = state.buffer.head()
        state.arcs.add(bufferFront, stackTop)
        
    def isCorrect(self, state, tree):
        if not self.isValid(state):
            return False
        
        stackTop = state.stack.top()
        bufferFront = state.buffer.head()
        return tree.hasArc(bufferFront, stackTop) and state.arcs.theSameChilden(stackTop, tree)
    
    def isValid(self, state):
        if state.stack.empty():
            return False

        return state.stack.length() == 1 or state.buffer.head() != tdatatypes.State.ROOT
    
class RightArc(tdatatypes.Transition):
    
    def _buildArc(self, state):
        stackTop = state.stack.top()
        stackSecond = state.getStackElem(1)
        return (stackSecond, stackTop)
    
    def update(self, state):
        stackTop = state.stack.pop()
        stackSecond = state.stack.top()
        state.arcs.add(stackSecond, stackTop)
    
    def isCorrect(self, state, tree):
        if not self.isValid(state):
            return False
        
        stackTop = state.stack.top()
        stackSecond = state.getStackElem(1)
        return tree.hasArc(stackSecond, stackTop) and state.arcs.theSameChilden(stackTop, tree)
    
    def isValid(self, state):
        if state.stack.length() < 2:
            return False
        
        return state.stack.top() != tdatatypes.State.ROOT
    
class Shift(tdatatypes.Transition):
    
    def _buildArc(self, _):
        return None
    
    def update(self, state):
        frontBuffer = state.buffer.pop()
        state.stack.push(frontBuffer)
  
    def isCorrect(self, state, _):
        return self.isValid(state)

    def isValid(self, state):
        return not state.buffer.empty() and state.buffer.head() != tdatatypes.State.ROOT
        
class Swap(tdatatypes.Transition):
    
    def _buildArc(self, _):
        return None
    
    def update(self, state):
        stackTop = state.stack.pop()
        bufferFront = state.buffer.pop()
        state.buffer.addFront(stackTop)
        state.buffer.addFront(bufferFront)
        
    def isCorrect(self, state, tree):
        if not self.isValid(state):
            return False
        
        stackTop = state.stack.top()
        bufferFront = state.buffer.head()
        return tree.areInOrder(bufferFront, stackTop)
    
    def isValid(self, state):
        if state.stack.length() < 1:
            return False
        
        stackTop = state.stack.top()
        bufferFront = state.buffer.head()
        return state.buffer.length() > 1 and stackTop < bufferFront
    
class ArcHybridWithSwap(tdatatypes.TransSystem):
    """This is the re-implementation of UU ArcHybrid which has ROOT at the end of the buffer. It can not re-use the archybrid transitions."""
    
    SHIFT = 0
    LEFTARC = 1
    RIGHTARC = 2
    SWAP = 3
       
    __SHIFT = Shift()
    __LEFTARC = LeftArc()
    __RIGHTARC = RightArc()
    __SWAP = Swap()
     
    def applyTransition(self, state, tId):
        self.__getTransition(tId).update(state)
    
    def isCorrectTransition(self, state, tree, tId):
        return self.__getTransition(tId).isCorrect(state, tree)
    
    def isValidTransition(self, state, tId):
        return self.__getTransition(tId).isValid(state)
    
    def isFinal(self, state):
        return state.stack.empty() and state.buffer.length() == 1 and state.buffer.head() == tdatatypes.State.ROOT
    
    def getNrOfTransitions(self):
        return 4
    
    def initialState(self, nrOfTokens):
        return tdatatypes.State(nrOfTokens, False)

    # functions for labeler
    def _buildArc(self, tId, state):
        return self.__getTransition(tId)._buildArc(state)
        
    def _getNonArcTransitions(self):
        return [ self.SHIFT, self.SWAP ] 
    
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
        elif tId == self.SWAP:
            return self.__SWAP
        else:
            raise RuntimeError("Unknown transition id: %i" % tId)
    
class ArcHybridWithSwapStaticOracle(oracle.Oracle):
    """Static oracle for ArcHybridWithSwap"""
    
    def __init__(self, system, labeler = None):
        self.__logger = logging.getLogger(self.__class__.__name__)
        self.__system = system
        self.__labeler = labeler
        
    def handlesNonProjectiveTrees(self):
        return True
    
    def handlesExploration(self):
        return False
    
    def nextCorrectTransitions(self, state, tree):
        transId = self.__nextCorrectId(state, tree)
        return [ self._getLabeledTransition(transId, state, tree) ]
    
    def __nextCorrectId(self, state, tree):
        transOrder = [ self.__system.LEFTARC, self.__system.SWAP, self.__system.RIGHTARC, self.__system.SHIFT ]
        
        for transId in transOrder:
            if self.__system.isCorrectTransition(state, tree, transId):
                return transId
            
        self.__logger.warning("No transition is correct in state: %s" % str(state))
        return None

    def _getLabeledTransition(self, tId, state, tree):
        if self.__labeler == None:
            return tId 
        else:
            if tId in [ self.__system.SHIFT, self.__system.SWAP ]:
                return self.__labeler.getLblTrans(tId)
            else:
                tArc = self.__system._buildArc(tId, state)
                lbl = tree.getLabel(tArc[1])
                return self.__labeler.getLblTrans(tId, lbl)
