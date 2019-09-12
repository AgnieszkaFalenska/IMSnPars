'''
Created on 18.08.2017

@author: falensaa
'''

import logging
import numpy as np

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
        return tree.hasArc(bufferFront, stackTop)
    
    def isValid(self, state):
        if state.stack.empty() or state.buffer.empty():
            return False
        
        stackTop = state.stack.top()
        return stackTop != tdatatypes.State.ROOT
    
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
        return not state.buffer.empty() 
        
class ArcHybrid(tdatatypes.TransSystem):
    """This is the normal ArcHybrid with root on the stack."""

    SHIFT = 0
    LEFTARC = 1
    RIGHTARC = 2
       
    __SHIFT = Shift()
    __LEFTARC = LeftArc()
    __RIGHTARC = RightArc()
    
    def applyTransition(self, state, tId):
        self.__getTransition(tId).update(state)
    
    def isCorrectTransition(self, state, tree, tId):
        return self.__getTransition(tId).isCorrect(state, tree)
    
    def isValidTransition(self, state, tId):
        return self.__getTransition(tId).isValid(state)

    def isFinal(self, state):
        return state.buffer.empty() and state.stack.length() == 1
    
    def getNrOfTransitions(self):
        return 3
    
    def initialState(self, nrOfTokens):
        return tdatatypes.State(nrOfTokens, True)

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
    

class ArcHybridStaticOracle(oracle.Oracle):
    """Static oracle for ArcHybrid"""
    
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
        
        transOrder = [ self.__system.LEFTARC, self.__system.RIGHTARC, self.__system.SHIFT ]
        
        for transId in transOrder:
            if self.__system.isCorrectTransition(state, tree, transId):
                return transId
        
        return None
    
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
            
class ArcHybridDynamicOracle(oracle.Oracle):
    def __init__(self, system, labeler, policy):
        self.__system = system
        self.__labeler = labeler
        self.__policy = policy

    def handlesNonProjectiveTrees(self):
        return False
    
    def handlesExploration(self):
        return True
    
    def doExploration(self, correctVal, predictedVal, epoch = 1):
        return not self.__policy.doCorrect(correctVal, predictedVal, epoch)
    
    def nextCorrectTransitions(self, state, tree):
        leftCost = self._calculateLeftCost(state, tree)
        rightCost = self._calculateRightCost(state, tree)
        shiftCost = self._calculateShiftCost(state, tree)
        
        result = [ ]
        
        if leftCost == 0:
            result.append(self.__system.LEFTARC)
        
        if rightCost == 0:
            result.append(self.__system.RIGHTARC)
        
        if shiftCost == 0:
            result.append(self.__system.SHIFT)
            
        return self._getLabeledTransitions(state, tree, result)
            
    def _calculateLeftCost(self, state, tree):
        leftCost = 0
        
        if state.stack.empty() or state.buffer.empty() or state.stack.top() == tdatatypes.State.ROOT:
            leftCost = np.inf
        else:
            # s0 children
            s0 = state.stack.top()
            leftCost += sum([ 1 if tree.hasArc(s0, d) else 0 for d in state.buffer ])
            
            # s0 head
            if state.stack.length() > 1:
                s1 = state.getStackElem(1)
                leftCost += 1 if tree.hasArc(s1, s0) else 0
            
            leftCost += sum([ 1 if tree.hasArc(h, s0) else 0 for h in state.buffer.tail() ])
            
        return leftCost
    
    def _calculateRightCost(self, state, tree):
        rightCost = 0
        
        if state.stack.length() < 2 or state.stack.top() == tdatatypes.State.ROOT:
            rightCost = np.inf
        else:
            # s0 children
            s0 = state.stack.top()
            rightCost += sum([ 1 if tree.hasArc(s0, d) else 0 for d in state.buffer ])
            rightCost += sum([ 1 if tree.hasArc(h, s0) else 0 for h in state.buffer ])
            
        return rightCost
    
    def _calculateShiftCost(self, state, tree):
        shiftCost = 0
        
        if state.buffer.empty():
            shiftCost = np.inf
        else:
            b = state.buffer.head()
            shiftCost += sum([ 1 if tree.hasArc(b, d) else 0 for d in state.stack ])
            
            if not state.stack.empty():
                shiftCost += sum([ 1 if tree.hasArc(h, b) else 0 for h in state.stack.tail() ])
        
        return shiftCost
         
    def _getLabeledTransitions(self, state, tree, tIds):
        if self.__labeler == None:
            return tIds
        
        result = [ ]
        for sysTrans in tIds:
            if sysTrans == self.__system.SHIFT:
                result.append(self.__labeler.getLblTrans(self.__system.SHIFT))
            else:
                tArc = self.__system._buildArc(sysTrans, state)
                if not tree.hasArc(tArc[0], tArc[1]):
                    result += self.__labeler.getAllLblTrans(sysTrans)
                else:
                    lbl = tree.getLabel(tArc[1])
                    result.append(self.__labeler.getLblTrans(sysTrans, lbl))    
        
        return result
    
    

