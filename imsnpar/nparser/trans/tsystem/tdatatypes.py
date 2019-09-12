'''
Created on 18.08.2017

@author: falensaa
'''

import abc
import heapq

from tools import datatypes

class Transition(object):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self):
        pass
    
    @abc.abstractmethod
    def update(self, state):
        """Applies transition to the parser state"""
        return

    @abc.abstractmethod
    def isValid(self, state):
        """Checks if the transition is valid for a given state"""
        return
    
    @abc.abstractmethod
    def _buildArc(self, state):
        return
        
class TransSystem:
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def getNrOfTransitions(self):
        """Returns nr of transitions"""
        return    
    
    @abc.abstractmethod
    def isFinal(self, state):
        """Checks if the configuration is final"""
        return
    
    @abc.abstractmethod
    def applyTransition(self, state, transId):
        return
    
    @abc.abstractmethod
    def isValidTransition(self, state, transId):
        return
    
    # functions for labeler
    
    @abc.abstractmethod
    def _buildArc(self, tId, state):
        return
        
        
    @abc.abstractmethod
    def _getNonArcTransitions(self):
        return 
    
    @abc.abstractmethod
    def _getArcTransitions(self):
        return
    

#TODO: convert this into mutable and not mutable
#TODO: build from simple array
class TreeUnderConstruction:
    NO_HEAD = datatypes.Tree.NO_HEAD
    NO_CHILDREN = 0
    
    def __init__(self, nrOfTokens):
        # tokPos -> headPos
        self.arcs = [ TreeUnderConstruction.NO_HEAD ] * (nrOfTokens)
        self.labels = [ None ] * (nrOfTokens)
  
        # tokenPos -> children
        self.children = [ [] for _ in range(nrOfTokens + 1) ]

    def getHead(self, tokPos):
        return self.arcs[tokPos]
    
    def getLabel(self, tokPos):
        return self.labels[tokPos]
    
    def hasHead(self, tokPos):
        return self.arcs[tokPos] != TreeUnderConstruction.NO_HEAD
  
    def add(self, headPos, depPos):
        self.arcs[depPos] = headPos
        heapq.heappush(self.children[headPos], depPos)
  
    def addLabel(self, arc, label):
        self.labels[arc[1]] = label
        
    #TODO: Slow function - should not be used in this form
    def getNrOfArcs(self):
        return len([h for h in self.arcs if h != TreeUnderConstruction.NO_HEAD])
    
    def hasArc(self, headPos, depPos):
        return self.arcs[depPos] == headPos
  
    def howManyChildren(self, headPos):
        return len(self.children[headPos])
    
    def __str__(self):
        return str(self.arcs)
    
    def buildTree(self):
        return datatypes.Tree(self.arcs, self.labels)
  
    #TODO:  It does not check exactly the children but should be enough
    def theSameChilden(self, tokenPos, tree):
        return tree.howManyChildren(tokenPos) == self.howManyChildren(tokenPos) 

    def getLeftChild(self, headPos, pos=0):
        if pos >= len(self.children[headPos]):
            return None
        
        fromLeft = heapq.nsmallest(pos+1, self.children[headPos])
        return fromLeft[pos]
    
    def getRightChild(self, headPos, pos=0):
        if pos >= len(self.children[headPos]):
            return None
        
        fromRight = heapq.nlargest(pos+1, self.children[headPos])
        return fromRight[pos]
    
    def getChildren(self, headPos):
        return self.children[headPos]
    
    
class State(object):
    ROOT = -1
    
    def __init__(self, nrOfTokens, putRootOnStack = True):
        self.nrOfTokens = nrOfTokens
  
        if putRootOnStack:
            # only root on the stack 
            self.stack = datatypes.Stack([ State.ROOT ])
  
            # all tokens in the buffer
            self.buffer = datatypes.Buffer(range(0, nrOfTokens))
  
        else:
            # the stack is empty 
            self.stack = datatypes.Stack([  ])
  
            # all tokens in the buffer
            self.buffer = datatypes.Buffer(range(0, nrOfTokens))
            
            # root at the end
            self.buffer.append(State.ROOT)
            
        # no arcs
        self.arcs = TreeUnderConstruction(nrOfTokens)

    def getStackElem(self, n):
        """Returns nth element on the stack"""
        
        if self.stack.length() > n:
            return self.stack.elem(n)
        else:
            return None
        
    def getBufferElem(self, n):
        """Returns nth element on the buffer"""
        
        if self.buffer.length() > n:
            return self.buffer.elem(n)
        else:
            return None
        
    def getLeftMostChild(self, headPos):
        if headPos == None:
            return None
        
        return self.arcs.getLeftChild(headPos)
    
    def getRightMostChild(self, headPos):
        if headPos == None:
            return None
        
        return self.arcs.getRightChild(headPos)
    
    def getLeftSecondChild(self, headPos):
        if headPos == None:
            return None
        
        return self.arcs.getLeftChild(headPos, 1)
    
    def getRightSecondChild(self, headPos):
        if headPos == None:
            return None
        
        return self.arcs.getRightChild(headPos, 1)
    
    def __str__(self):
        return "%s\t%s\t%s" % (self.stack, self.buffer, self.arcs)
    
def buildTreeFromTransitions(system, nrOfTokens, transitions):
    state = system.initialState(nrOfTokens)
    for trans in transitions:
        system.applyTransition(state, trans)
            
    return state.arcs.buildTree()
