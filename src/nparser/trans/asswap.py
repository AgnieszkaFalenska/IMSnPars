'''
Created on 18.08.2017

@author: falensaa
'''

import logging

from tools import datatypes
from nparser.trans.tsystem import Transition, TransSystem, State
from nparser.trans import arcstandard, tsystem
from nparser.trans import  oracle
from nparser.trans.arcstandard import ArcStandard, ArcStandardStaticOracle

class Swap(Transition):
    
    def _buildArc(self, _):
        return None
    
    def update(self, state):
        stackTop = state.stack.pop()
        stackSecond = state.stack.pop()
        state.stack.push(stackTop)
        state.buffer.addFront(stackSecond)
    
    def isCorrect(self, state, tree):
        if not self.isValid(state):
            return False
        
        stackTop = state.getStackElem(0)
        stackSecond = state.getStackElem(1)
        return tree.areInOrder(stackTop, stackSecond)

    def isValid(self, state):
        if state.stack.length() < 2:
            return False
        
        stackTop = state.getStackElem(0)
        stackSecond = state.getStackElem(1)
        return stackSecond != State.ROOT and stackTop > stackSecond
    
class ArcStandardWithSwap(TransSystem):
    
    SHIFT = 0
    LEFTARC = 1
    RIGHTARC = 2
    SWAP = 3
       
    __SHIFT = arcstandard.Shift()
    __LEFTARC = arcstandard.LeftArc()
    __RIGHTARC = arcstandard.RightArc()
    __SWAP = Swap()
     
    def applyTransition(self, state, tId):
        self.__getTransition(tId).update(state)
    
    def isCorrectTransition(self, state, tree, tId):
        return self.__getTransition(tId).isCorrect(state, tree)
    
    def isValidTransition(self, state, tId):
        return self.__getTransition(tId).isValid(state)

    def isFinal(self, state):
        return state.buffer.empty() and state.stack.length() == 1
    
    def initialState(self, nrOfTokens):
        return State(nrOfTokens, True)
    
    def getNrOfTransitions(self):
        return 4
    
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
        
        
class ArcStandardWithSwapEagerOracle(oracle.Oracle):
    """Static eager oracle for ArcStandardWithSwap"""
    
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
        if transId == None:
            return [ None ]
        
        return [ self._getLabeledTransition(transId, state, tree) ]
    
    def __nextCorrectId(self, state, tree):
        if self.__canLA(state, tree):
            return self.__system.LEFTARC
        elif self.__canRA(state, tree):
            return self.__system.RIGHTARC
        elif self.__canSW(state, tree):
            return self.__system.SWAP
        elif self.__canSH(state, tree):
            return self.__system.SHIFT
        
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

    def __canSW(self, state, tree):
        if not self.__system.isValidTransition(state, self.__system.SWAP):
            return False
        
        head, dep = self.__system._buildArc(self.__system.LEFTARC, state)
        return tree.areInOrder(head, dep)
    
class MPCBuilder(object):
    def __init__(self):
        self.__noSwap = ArcStandard()
        self.__noSwapOracle = ArcStandardStaticOracle(self.__noSwap)
        
    def buildMPCs(self, tree):
        noSwapTrans = oracle.buildStaticCorrectTransitions(tree, self.__noSwap, self.__noSwapOracle)
        projectiveForest = tsystem.buildTreeFromTransitions(self.__noSwap, tree.nrOfTokens(), noSwapTrans)
        #print "projectiveForest", projectiveForest
        return self.__buildMPCsForTree(projectiveForest)
            
    def __buildMPCsForTree(self, tree):
        def updateMPCs(mpc, head, transChildren, mpcs):
            mpcs[head] = mpc
            
            if head not in transChildren:
                return
            
            for child in transChildren[head]:
                updateMPCs(mpc, child, transChildren, mpcs)
            
        transitiveChildren = tree.buildTransitiveChildren()
        
        if not datatypes.Tree.NO_HEAD in transitiveChildren:
            return { i : datatypes.Tree.ROOT for i in range(tree.nrOfTokens()) }
        
        mpcs = { head : head for head in transitiveChildren[datatypes.Tree.NO_HEAD] }
        
        for node in transitiveChildren[datatypes.Tree.NO_HEAD]:
            updateMPCs(node, node, transitiveChildren, mpcs)
        
        return mpcs
        
class ArcStandardWithSwapLazyOracle(oracle.Oracle):
    """Static lazy oracle for ArcStandardWithSwap"""
    
    def __init__(self, system, labeler = None):
        self.__logger = logging.getLogger(self.__class__.__name__)
        self.__system = system
        self.__labeler = labeler
        self.__mpcs = MPCBuilder()
        
    def handlesNonProjectiveTrees(self):
        return True
    
    def handlesExploration(self):
        return False
    
    def nextCorrectTransitions(self, state, tree):
        transId = self.__nextCorrectId(state, tree)
        if transId == None:
            return [ None ]
        
        return [ self._getLabeledTransition(transId, state, tree) ]
    
    def __nextCorrectId(self, state, tree):
        # bleee this is super ugly
        if not hasattr(tree, "mpcs"):
            tree.mpcs = self.__mpcs.buildMPCs(tree)
            
        if self.__canLA(state, tree):
            return self.__system.LEFTARC
        elif self.__canRA(state, tree):
            return self.__system.RIGHTARC
        elif self.__canSW(state, tree):
            return self.__system.SWAP
        elif self.__canSH(state, tree):
            return self.__system.SHIFT
        
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

    def __canSW(self, state, tree):
        if not self.__system.isValidTransition(state, self.__system.SWAP):
            return False    
        
        stackTop = state.stack.top()
        stackSecond = state.getStackElem(1)
        if not tree.areInOrder(stackTop, stackSecond):
            return False
        
        if state.buffer.empty():
            return True
        
        bufferFront = state.buffer.head()
        return tree.mpcs[stackTop] != tree.mpcs[bufferFront] 
    