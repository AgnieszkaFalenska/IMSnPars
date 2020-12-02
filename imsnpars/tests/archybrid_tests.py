#!/usr/bin/env python
# coding=utf-8

'''
Created on 21.08.2017

@author: falensaa
'''

import unittest
from nparser.trans.tsystem.archybrid import ArcHybrid, ArcHybridStaticOracle
from nparser.trans.tsystem import oracle, tdatatypes
from tests.examples import *

class TestTransitionMethods(unittest.TestCase):

    def testLeftArcHead(self):
        """LeftArc introduces an arc from the front of the buffer to the top-most token on the stack"""
        
        state = tdatatypes.State(5)
        system = ArcHybrid()
        system.applyTransition(state, ArcHybrid.SHIFT)
        system.applyTransition(state, ArcHybrid.LEFTARC)
        self.assertEqual(state.arcs.getHead(0), 1)
    
    def testLeftArcRemove(self):
        """LeftArc removes the top-most token on the stack"""
        
        state = tdatatypes.State(5)
        system = ArcHybrid()
        system.applyTransition(state, ArcHybrid.SHIFT)
        
        self.assertEqual(state.stack.toList(), [ -1, 0 ])
        system.applyTransition(state, ArcHybrid.LEFTARC)
        self.assertEqual(state.stack.toList(), [ -1 ])
    
    def testRightArcHead(self):
        """RightArcStandard introduces an arc from the second item to the top-most token on the stack"""
        
        state = tdatatypes.State(5)
        system = ArcHybrid()
        system.applyTransition(state, ArcHybrid.SHIFT)
        
        system.applyTransition(state, ArcHybrid.RIGHTARC)
        self.assertEqual(state.arcs.getHead(0), -1)
    
    def testRightArcRemove(self):
        """RightArcStandard removes the top-most token from the stack"""
        
        state = tdatatypes.State(5)
        system = ArcHybrid()
        system.applyTransition(state, ArcHybrid.SHIFT)
        
        self.assertEqual(state.stack.toList(), [ -1, 0 ])
        self.assertEqual(state.buffer.toList(), [ 1, 2, 3, 4 ])
        
        system.applyTransition(state, ArcHybrid.RIGHTARC)
        
        self.assertEqual(state.stack.toList(), [ -1 ])
        self.assertEqual(state.buffer.toList(), [ 1, 2, 3, 4 ])
    
    def testShift(self):
        """Shift takes the first token from the front of the buffer and pushes it onto the stack"""
        
        state = tdatatypes.State(5)
        system = ArcHybrid()
        
        system.applyTransition(state, ArcHybrid.SHIFT)
        
        self.assertEqual(state.buffer.toList(), [ 1, 2, 3, 4 ])
        self.assertEqual(state.stack.toList(), [ -1, 0 ])

oneWordCorrect = [ ArcHybrid.SHIFT, ArcHybrid.RIGHTARC ]

msCollinsCorrect = [ ArcHybrid.SHIFT, ArcHybrid.LEFTARC, ArcHybrid.SHIFT, ArcHybrid.SHIFT, ArcHybrid.RIGHTARC, ArcHybrid.SHIFT, ArcHybrid.SHIFT, ArcHybrid.LEFTARC, ArcHybrid.SHIFT, ArcHybrid.LEFTARC, 
                     ArcHybrid.SHIFT, ArcHybrid.SHIFT, ArcHybrid.RIGHTARC, ArcHybrid.RIGHTARC, ArcHybrid.RIGHTARC, ArcHybrid.SHIFT, ArcHybrid.RIGHTARC, ArcHybrid.RIGHTARC ]

franceCorrect = [ArcHybrid.SHIFT, ArcHybrid.SHIFT, ArcHybrid.RIGHTARC, ArcHybrid.SHIFT, ArcHybrid.RIGHTARC, ArcHybrid.SHIFT, ArcHybrid.RIGHTARC, ArcHybrid.SHIFT, ArcHybrid.RIGHTARC, ArcHybrid.SHIFT,
                  ArcHybrid.RIGHTARC, ArcHybrid.RIGHTARC ]

chamberMusicCorrect = [ArcHybrid.SHIFT, ArcHybrid.SHIFT, ArcHybrid.RIGHTARC, ArcHybrid.SHIFT, ArcHybrid.LEFTARC, ArcHybrid.SHIFT, ArcHybrid.SHIFT, ArcHybrid.SHIFT, ArcHybrid.LEFTARC, ArcHybrid.SHIFT,
                        ArcHybrid.RIGHTARC, ArcHybrid.RIGHTARC, ArcHybrid.RIGHTARC, ArcHybrid.SHIFT, ArcHybrid.RIGHTARC, ArcHybrid.RIGHTARC]

changesCorrect = [ArcHybrid.SHIFT, ArcHybrid.LEFTARC, ArcHybrid.SHIFT, ArcHybrid.SHIFT, ArcHybrid.RIGHTARC, ArcHybrid.SHIFT, ArcHybrid.LEFTARC, ArcHybrid.SHIFT, ArcHybrid.RIGHTARC, 
                   ArcHybrid.LEFTARC, ArcHybrid.SHIFT, ArcHybrid.SHIFT, ArcHybrid.LEFTARC, ArcHybrid.SHIFT, ArcHybrid.RIGHTARC, ArcHybrid.SHIFT, ArcHybrid.RIGHTARC, ArcHybrid.RIGHTARC]

class TestBuildTreeFromArcHybridTransitions(unittest.TestCase):
     
    def testOneWordExample(self):
        """Example: oneWordExample"""
     
        system = ArcHybrid()
        tree = tdatatypes.buildTreeFromTransitions(system, oneWordExample[1].nrOfTokens(), oneWordCorrect)
        self.assertEqual(tree, oneWordExample[1])
 
    def testMsCollinsExample(self):
        """Example: msCollinsExample"""
     
        system = ArcHybrid()
        tree = tdatatypes.buildTreeFromTransitions(system, msCollinsExample[1].nrOfTokens(), msCollinsCorrect)
        self.assertEqual(tree, msCollinsExample[1])
         
    def testFranceExample(self):
        """Example: franceExample"""
      
        system = ArcHybrid()
        tree = tdatatypes.buildTreeFromTransitions(system, franceExample[1].nrOfTokens(), franceCorrect)
        self.assertEqual(tree, franceExample[1])
          
    def testChamberMusicExample(self):
        """Example: chamberMusicExample"""
      
        system = ArcHybrid()
        tree = tdatatypes.buildTreeFromTransitions(system, chamberMusicExample[1].nrOfTokens(), chamberMusicCorrect)
        self.assertEqual(tree, chamberMusicExample[1])
          
    def testChangesExample(self):
        """Example: changesExample"""
      
        system = ArcHybrid()
        tree = tdatatypes.buildTreeFromTransitions(system, changesExample[1].nrOfTokens(), changesCorrect)
        self.assertEqual(tree, changesExample[1])
         
 
class TestArcHybridStaticOracle(unittest.TestCase):
    
    def testOneWordExample(self):
        """Example: oneWordExample"""
    
        system = ArcHybrid()
        soracle = ArcHybridStaticOracle(system)
        transitions = oracle.buildStaticCorrectTransitions(oneWordExample[1], system, soracle)
        self.assertEqual(oneWordCorrect, transitions)

    def testMsCollinsExample(self):
        """Example: msCollinsExample"""
    
        system = ArcHybrid()
        soracle = ArcHybridStaticOracle(system)
        transitions = oracle.buildStaticCorrectTransitions(msCollinsExample[1], system, soracle)
        self.assertEqual(msCollinsCorrect, transitions)
        
    def testFranceExample(self):
        """Example: franceExample"""
    
        system = ArcHybrid()
        soracle = ArcHybridStaticOracle(system)
        transitions = oracle.buildStaticCorrectTransitions(franceExample[1], system, soracle)
        self.assertEqual(franceCorrect, transitions)
        
    def testChamberMusicExample(self):
        """Example: chamberMusicExample"""
    
        system = ArcHybrid()
        soracle = ArcHybridStaticOracle(system)
        transitions = oracle.buildStaticCorrectTransitions(chamberMusicExample[1], system, soracle)
        self.assertEqual(chamberMusicCorrect, transitions)
        
    def testChangesExample(self):
        """Example: changesExample"""
    
        system = ArcHybrid()
        soracle = ArcHybridStaticOracle(system)
        transitions = oracle.buildStaticCorrectTransitions(changesExample[1], system, soracle)
        self.assertEqual(changesCorrect, transitions)
         
if __name__ == "__main__":
    unittest.main()