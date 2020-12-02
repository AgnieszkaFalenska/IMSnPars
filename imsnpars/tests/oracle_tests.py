#!/usr/bin/env python
# coding=utf-8

'''
Created on 21.08.2017

@author: falensaa
'''

import unittest
from nparser.trans.tsystem.archybrid import ArcHybrid, ArcHybridStaticOracle, ArcHybridDynamicOracle
from nparser.trans.tsystem import oracle, tdatatypes
from tests.archybrid_tests import oneWordCorrect, msCollinsCorrect, franceCorrect, chamberMusicCorrect, changesCorrect
from tests.examples import *

class MockExplorePolicy(oracle.ExplorePolicy):
    def __init__(self):
        pass
    
    def doCorrect(self, correctVal, predictedVal, epoch = 1):
        return True
    
class TestHybridDynamicOracleMethods(unittest.TestCase):
    
    def testOneWordExample_correctPath(self):
        """Example: oneWordExample"""
     
        transitions = self.__buildCorrectTransitions(oneWordExample[1])
        self.assertEqual(oneWordCorrect, transitions)
 
    def testMsCollinsExample_correctPath(self):
        """Example: msCollinsExample"""
     
        transitions = self.__buildCorrectTransitions(msCollinsExample[1])
        self.assertEqual(msCollinsCorrect, transitions)
         
    def testFranceExample_correctPath(self):
        """Example: franceExample"""
     
        transitions = self.__buildCorrectTransitions(franceExample[1])
        self.assertEqual(franceCorrect, transitions)
         
    def testChamberMusicExample_correctPath(self):
        """Example: chamberMusicExample"""
     
        transitions = self.__buildCorrectTransitions(chamberMusicExample[1])
        self.assertEqual(chamberMusicCorrect, transitions)
         
    def testChangesExample_correctPath(self):
        """Example: changesExample"""
     
        transitions = self.__buildCorrectTransitions(changesExample[1])
        self.assertEqual(changesCorrect, transitions)
     
    def __buildCorrectTransitions(self, tree):
        system = ArcHybrid()
        soracle =  ArcHybridDynamicOracle(system, None, MockExplorePolicy())
        
        result = [ ]
        state = tdatatypes.State(tree.nrOfTokens())
        
        while not system.isFinal(state):
            transition = soracle.nextCorrectTransitions(state, tree)[0]
            result.append(transition)
            system.applyTransition(state, transition)
        
        return result

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()