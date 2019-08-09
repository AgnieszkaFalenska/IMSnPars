'''
Created on 25.08.2017

@author: falensaa
'''

import logging, pickle
from nparser.trans.tsystem import TransSystem
from tools.nn_tools import NNTreeTask

class TransSystemLabeler(TransSystem):
    
    def __init__(self, system):
        self.__logger = logging.getLogger(self.__class__.__name__)
        
        self.__system = system
        self.__idToSysTrans = None
        self.__sysTransToId = None
    
    # trans system functions
    
    def isFinal(self, state):
        return self.__system.isFinal(state)
    
    def initialState(self, nrOfTokens):
        return self.__system.initialState(nrOfTokens)
    
    def getNrOfTransitions(self):
        return len(self.__idToSysTrans)
    
    def applyTransition(self, state, tId):
        transId, label = self.__idToSysTrans[tId]
        arc = self.__system._buildArc(transId, state)
        self.__system.applyTransition(state, transId)
        
        if (arc == None) != (label == None):
            self.__logger.warn("Error state: %i %i,%i %s" % (tId, arc[0], arc[1], label))
            
        assert((arc == None) == (label == None))
        
        if arc != None:
            state.arcs.addLabel(arc, label)
    
    def isValidTransition(self, state, tId):
        transId, _ = self.__idToSysTrans[tId]
        return self.__system.isValidTransition(state, transId)
    
    def isCorrectTransition(self, state, tree, tId):
        transId, lbl = self.__idToSysTrans[tId]
        if not self.__system.isCorrectTransition(state, tree, transId):
            return False
        
        arc = self.__system._buildArc(transId, state)
        if arc == None:
            return True
        
        (head, dep) = arc
        assert tree.hasArc(head, dep)
        return tree.getLabel(dep) == lbl
    
    ## labeler functions
    
    def readData(self, sentences):
        labels = set()
        for sent in sentences:
            for tok in sent:
                labels.add(tok.dep)
                
        self._storeLabels(labels)
        
    def save(self, pickleOut):
        pickle.dump((self.__idToSysTrans, self.__sysTransToId), pickleOut)
         
    def load(self, pickleIn):
        (self.__idToSysTrans, self.__sysTransToId) = pickle.load(pickleIn)

    def getLblTrans(self, sysTrans, lbl = None):
        return self.__sysTransToId[(sysTrans, lbl)]
    
    def getSysTrans(self, lblTrans):
        transId, _ = self.__idToSysTrans[lblTrans]
        return transId
        
    def getAllLblTrans(self, sysTrans):
        result = [ ]
        for (sTrans, _), lblTrans in self.__sysTransToId.items():
            if sTrans == sysTrans:
                result.append(lblTrans)
         
        return result
    
    def _storeLabels(self, labels):
        self.__sysTransToId = { }
        self.__idToSysTrans = { }
        
        for trans in self.__system._getNonArcTransitions():
            transId = len(self.__sysTransToId)
            self.__sysTransToId[(trans, None)] = transId
            self.__idToSysTrans[transId] = (trans, None)
            
        for trans in self.__system._getArcTransitions():
            for lbl in labels:
                transId = len(self.__sysTransToId)
                self.__sysTransToId[(trans, lbl)] = transId
                self.__idToSysTrans[transId] = (trans, lbl)
                
    # functions for labeler
    def _buildArc(self, tId, state):
        transId, _ = self.__idToSysTrans[tId]
        return self.__system._buildArc(transId, state)
        
    def _getNonArcTransitions(self):
        result = [ ]
        for tId, lbl in self.__idToSysTrans.items():
            if lbl == None:
                result.append(tId)
                
        return result
    
    def _getArcTransitions(self):
        result = [ ]
        for tId, lbl in self.__idToSysTrans.items():
            if lbl != None:
                result.append(tId)
                
        return result
    