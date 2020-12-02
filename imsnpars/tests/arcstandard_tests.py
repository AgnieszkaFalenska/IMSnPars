#!/usr/bin/env python
# coding=utf-8

'''
Created on 21.08.2017

@author: falensaa
'''

import unittest
from nparser.trans.tsystem.arcstandard import ArcStandard, ArcStandardStaticOracle
from nparser.trans.tsystem import oracle, tdatatypes
from tests.examples import *

class TestTransitionMethods(unittest.TestCase):

    def testLeftArcHead(self):
        """LeftArc introduces an arc from the second to the first item on the stack"""
        
        state = tdatatypes.State(5)
        system = ArcStandard()
        system.applyTransition(state, ArcStandard.SHIFT)
        system.applyTransition(state, ArcStandard.SHIFT)
        system.applyTransition(state, ArcStandard.LEFTARC)
        self.assertEqual(state.arcs.getHead(0), 1)
    
    def testLeftArcRemove(self):
        """LeftArc removes the second token on the stack"""
        
        state = tdatatypes.State(5)
        system = ArcStandard()
        system.applyTransition(state, ArcStandard.SHIFT)
        
        self.assertEqual(state.stack.toList(), [ -1, 0 ])
        system.applyTransition(state, ArcStandard.LEFTARC)
        self.assertEqual(state.stack.toList(), [ 0 ])
    
    def testRightArcHead(self):
        """RightArcStandard introduces an arc from the top-most token on the stack to the second one"""
        
        state = tdatatypes.State(5)
        system = ArcStandard()
        system.applyTransition(state, ArcStandard.SHIFT)
        system.applyTransition(state, ArcStandard.RIGHTARC)
        self.assertEqual(state.arcs.getHead(0), -1)
    
    def testRightArcRemove(self):
        """RightArcStandard removes the top-most token from the stack"""
        
        state = tdatatypes.State(5)
        system = ArcStandard()
        system.applyTransition(state, ArcStandard.SHIFT)
        
        self.assertEqual(state.stack.toList(), [ -1, 0 ])
        self.assertEqual(state.buffer.toList(), [ 1, 2, 3, 4 ])
        
        system.applyTransition(state, ArcStandard.RIGHTARC)
        
        self.assertEqual(state.stack.toList(), [ -1 ])
        self.assertEqual(state.buffer.toList(), [ 1, 2, 3, 4 ])
    
    def testShift(self):
        """Shift takes the first token from the front of the buffer and pushes it onto the stack"""
        
        state = tdatatypes.State(5)
        system = ArcStandard()
        
        system.applyTransition(state, ArcStandard.SHIFT)
        
        self.assertEqual(state.buffer.toList(), [ 1, 2, 3, 4 ])
        self.assertEqual(state.stack.toList(), [ -1, 0 ])

    def testLeftArcIsNotValidWithRoot(self):
        """invalid left when second item on the stack is root"""
        
        state = tdatatypes.State(5)
        system = ArcStandard()
        system.applyTransition(state, ArcStandard.SHIFT)       
        self.assertFalse(system.isValidTransition(state, ArcStandard.LEFTARC))
      
    def testFinalConfigurationBufferNonEmpty(self):
        """configuration with full buffer is not final"""
        
        state = tdatatypes.State(5)
        system = ArcStandard()
        self.assertFalse(system.isFinal(state))
      
    def testFinalConfigurationBufferEmpty(self):
        """final configuration - with empty buffer"""
        
        state = tdatatypes.State(0)
        system = ArcStandard()
        self.assertTrue(system.isFinal(state))
        
oneWordCorrect = [ ArcStandard.SHIFT, ArcStandard.RIGHTARC ]

msCollinsCorrect = [ArcStandard.SHIFT, ArcStandard.SHIFT, ArcStandard.LEFTARC, ArcStandard.SHIFT, ArcStandard.RIGHTARC, ArcStandard.SHIFT, ArcStandard.SHIFT, ArcStandard.SHIFT, ArcStandard.LEFTARC, ArcStandard.SHIFT, ArcStandard.LEFTARC, ArcStandard.SHIFT, ArcStandard.RIGHTARC, ArcStandard.RIGHTARC, ArcStandard.RIGHTARC, ArcStandard.SHIFT, ArcStandard.RIGHTARC, ArcStandard.RIGHTARC]

franceCorrect = [ArcStandard.SHIFT, ArcStandard.SHIFT, ArcStandard.RIGHTARC, ArcStandard.SHIFT, ArcStandard.RIGHTARC, ArcStandard.SHIFT, ArcStandard.RIGHTARC, ArcStandard.SHIFT, ArcStandard.RIGHTARC, ArcStandard.SHIFT, ArcStandard.RIGHTARC, ArcStandard.RIGHTARC]

chamberMusicCorrect = [ArcStandard.SHIFT, ArcStandard.SHIFT, ArcStandard.RIGHTARC, ArcStandard.SHIFT, ArcStandard.SHIFT, ArcStandard.LEFTARC, ArcStandard.SHIFT, ArcStandard.SHIFT, ArcStandard.SHIFT, ArcStandard.LEFTARC, ArcStandard.RIGHTARC, ArcStandard.RIGHTARC, ArcStandard.RIGHTARC, ArcStandard.SHIFT, ArcStandard.RIGHTARC, ArcStandard.RIGHTARC] 

[ArcStandard.SHIFT, ArcStandard.RIGHTARC, ArcStandard.SHIFT, ArcStandard.SHIFT, ArcStandard.LEFTARC, ArcStandard.SHIFT, ArcStandard.SHIFT, ArcStandard.SHIFT, ArcStandard.LEFTARC, ArcStandard.RIGHTARC,
                        ArcStandard.RIGHTARC, ArcStandard.RIGHTARC, ArcStandard.SHIFT, ArcStandard.RIGHTARC, ArcStandard.RIGHTARC, ArcStandard.SHIFT]

changesCorrect =  [ArcStandard.SHIFT, ArcStandard.SHIFT, ArcStandard.LEFTARC, ArcStandard.SHIFT, ArcStandard.RIGHTARC, ArcStandard.SHIFT, ArcStandard.SHIFT, ArcStandard.LEFTARC, ArcStandard.RIGHTARC, ArcStandard.SHIFT, ArcStandard.LEFTARC, ArcStandard.SHIFT, ArcStandard.SHIFT, ArcStandard.LEFTARC, ArcStandard.RIGHTARC, ArcStandard.SHIFT, ArcStandard.RIGHTARC, ArcStandard.RIGHTARC]
    
class TestArcStandardOracle(unittest.TestCase):
    
    def testOneWordExample(self):
        """Example: oneWordExample"""
    
        system = ArcStandard()
        soracle = ArcStandardStaticOracle(system)
        transitions = oracle.buildStaticCorrectTransitions(oneWordExample[1], system, soracle)
        self.assertEqual(oneWordCorrect, transitions)

    def testMsCollinsExample(self):
        """Example: msCollinsExample"""
    
        system = ArcStandard()
        soracle = ArcStandardStaticOracle(system)
        transitions = oracle.buildStaticCorrectTransitions(msCollinsExample[1], system, soracle)
        self.assertEqual(msCollinsCorrect, transitions)
        
    def testFranceExample(self):
        """Example: franceExample"""
    
        system = ArcStandard()
        soracle = ArcStandardStaticOracle(system)
        transitions = oracle.buildStaticCorrectTransitions(franceExample[1], system, soracle)
        self.assertEqual(franceCorrect, transitions)
        
    def testChamberMusicExample(self):
        """Example: chamberMusicExample"""
    
        system = ArcStandard()
        soracle = ArcStandardStaticOracle(system)
        transitions = oracle.buildStaticCorrectTransitions(chamberMusicExample[1], system, soracle)
        self.assertEqual(chamberMusicCorrect, transitions)
        
    def testChangesExample(self):
        """Example: changesExample"""
    
        system = ArcStandard()
        soracle = ArcStandardStaticOracle(system)
        transitions = oracle.buildStaticCorrectTransitions(changesExample[1], system, soracle)
        self.assertEqual(changesCorrect, transitions)
        

class TestBuildTreeFromTransitions(unittest.TestCase):
    
    def testOneWordExample(self):
        """Example: oneWordExample"""
        
        system = ArcStandard()
        tree = tdatatypes.buildTreeFromTransitions(system, oneWordExample[1].nrOfTokens(), oneWordCorrect)
        self.assertEqual(tree, oneWordExample[1])

    def testMsCollinsExample(self):
        """Example: msCollinsExample"""
    
        system = ArcStandard()
        tree = tdatatypes.buildTreeFromTransitions(system, msCollinsExample[1].nrOfTokens(), msCollinsCorrect)
        self.assertEqual(tree, msCollinsExample[1])
        
    def testFranceExample(self):
        """Example: franceExample"""
    
        system = ArcStandard()
        tree = tdatatypes.buildTreeFromTransitions(system, franceExample[1].nrOfTokens(), franceCorrect)
        self.assertEqual(tree, franceExample[1])
        
    def testChamberMusicExample(self):
        """Example: chamberMusicExample"""
    
        system = ArcStandard()
        tree = tdatatypes.buildTreeFromTransitions(system, chamberMusicExample[1].nrOfTokens(), chamberMusicCorrect)
        self.assertEqual(tree, chamberMusicExample[1])
        
    def testChangesExample(self):
        """Example: changesExample"""
    
        system = ArcStandard()
        tree = tdatatypes.buildTreeFromTransitions(system, changesExample[1].nrOfTokens(), changesCorrect)
        self.assertEqual(tree, changesExample[1])
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()