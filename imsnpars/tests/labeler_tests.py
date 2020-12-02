#!/usr/bin/env python
# coding=utf-8

'''
Created on 25.08.2017

@author: falensaa
'''

import unittest
from tools import datatypes
from nparser.trans.tsystem import oracle
from nparser.trans.tsystem import tdatatypes
from nparser.trans.labeler import TransSystemLabeler
from nparser.trans.tsystem.arcstandard import ArcStandard, ArcStandardStaticOracle
from nparser.trans.tsystem.archybrid import ArcHybrid, ArcHybridStaticOracle

class FakeTransSystem(tdatatypes.TransSystem):
    def __init__(self, arcTrans, nonArcTrans):
        self.arcTrans = arcTrans
        self.nonArcTrans = nonArcTrans
        
    def getNrOfTransitions(self):
        return len(self.arcTrans) + len(self.nonArcTrans)
    
    def _getArcTransitions(self):
        return self.arcTrans
    
    def _getNonArcTransitions(self):
        return self.nonArcTrans
    
    def getNextPossibleTransition(self):
        return
    
    def nextTransition(self):
        return
    
    def isFinal(self, _):
        """Checks if the configuration is final"""
        return
    
    # NOT IMPLEMENTED
    
    def _buildArc(self, tId, state):
        return None
        
    def applyTransition(self, state, transId):
        return

    def isValidTransition(self, state, transId):
        return
    
class MockArcStandard(ArcStandard):
    def __init__(self):
        self.transitions = [ ]
    
    def applyTransition(self, state, tId):
        self.transitions.append(tId)
        return ArcStandard.applyTransition(self, state, tId)
        
class MockLabeler(TransSystemLabeler):
    def __init__(self, system, labels):
        TransSystemLabeler.__init__(self, system)
        self._storeLabels(labels)
        
class TestTransSystemMethods(unittest.TestCase):
    
    def testNrOfTransitions_noTrans(self):
        system = FakeTransSystem([ ], [ ])
        labeler = MockLabeler(system, [ "a", "b" ])
        self.assertEqual(labeler.getNrOfTransitions(), 0)
        
    def testNrOfTransitions_onlyArc(self):
        system = FakeTransSystem([ 1 ], [ ])
        labeler = MockLabeler(system, [ "a", "b" ])
        self.assertEqual(labeler.getNrOfTransitions(), 2)

    def testNrOfTransitions_onlyNonArc(self):
        system = FakeTransSystem([ ], [ 1 ])
        labeler = MockLabeler(system, [ "a", "b" ])
        self.assertEqual(labeler.getNrOfTransitions(), 1)
        
    def testNrOfTransitions_bothTypes(self):
        system = FakeTransSystem([ 0, 1, 2 ], [ 3, 4 ])
        labeler = MockLabeler(system, [ "a", "b" ])
        self.assertEqual(labeler.getNrOfTransitions(), 8)
        
    def testMimicSystemWithOneLabel_franceExample(self):
        correctTree = datatypes.Tree([-1, 0, 0, 0, 0, 0], [ "ROOT", "a", "b", "c", "d", "e"])
        correctTrans = [ArcStandard.SHIFT, ArcStandard.SHIFT, ArcStandard.RIGHTARC, ArcStandard.SHIFT, ArcStandard.RIGHTARC, ArcStandard.SHIFT, ArcStandard.RIGHTARC, ArcStandard.SHIFT, ArcStandard.RIGHTARC, 
                  ArcStandard.SHIFT, ArcStandard.RIGHTARC, ArcStandard.RIGHTARC ]
        
        mocksystem = MockArcStandard()
        labeler = MockLabeler(mocksystem, [ "ROOT", "a", "b", "c", "d", "e"])
        
        soracle = ArcStandardStaticOracle(mocksystem, labeler)
        labtransitions = oracle.buildStaticCorrectTransitions(correctTree, labeler, soracle)
        
        # internal transitions
        self.assertEqual(mocksystem.transitions, correctTrans)
        
        # applying labeled transitions
        predTree = tdatatypes.buildTreeFromTransitions(labeler, correctTree.nrOfTokens(), labtransitions)
        self.assertEqual(str(predTree), str(correctTree))
        
    def testArcStandardWithLabels(self):
        system = ArcStandard()
        labeler = MockLabeler(system, [ "ROOT", "a", "b" ])
         
        correct = datatypes.Tree( [ 1, -1, 1, 1, 5, 6, 3, 6, 1 ], [ "a", "ROOT", "a", "b", "b", "a", "a", "b", "a"])
        
        soracle = ArcStandardStaticOracle(system, labeler)
        transitions = oracle.buildStaticCorrectTransitions(correct, labeler, soracle)
        
        predict = tdatatypes.buildTreeFromTransitions(labeler, correct.nrOfTokens(), transitions)
        self.assertEqual(predict, correct)
        
    def testArcHybridWithLabels(self):
        system = ArcHybrid()
        labeler = MockLabeler(system, [ "ROOT", "a", "b" ])
         
        correct = datatypes.Tree( [ 1, -1, 1, 1, 5, 6, 3, 6, 1 ], [ "a", "ROOT", "a", "b", "b", "a", "a", "b", "a"])
        soracle = ArcHybridStaticOracle(system, labeler)
        transitions = oracle.buildStaticCorrectTransitions(correct, labeler, soracle)
        
        predict = tdatatypes.buildTreeFromTransitions(labeler, correct.nrOfTokens(), transitions)
        self.assertEqual(predict, correct)
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()