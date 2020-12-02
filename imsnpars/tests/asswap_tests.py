#!/usr/bin/env python
# coding=utf-8

'''
Created on 21.08.2017

@author: falensaa
'''

import unittest
from nparser.trans.tsystem.asswap import ArcStandardWithSwap as ASWS, MPCBuilder
from nparser.trans.tsystem.asswap import ArcStandardWithSwapEagerOracle, ArcStandardWithSwapLazyOracle
from nparser.trans.tsystem import oracle
from tests.examples import *

oneWordCorrect = [ ASWS.SHIFT, ASWS.RIGHTARC ]

msCollinsCorrect = [ ASWS.SHIFT, ASWS.SHIFT, ASWS.LEFTARC,  ASWS.SHIFT, ASWS.RIGHTARC,
                      ASWS.SHIFT, ASWS.SHIFT, ASWS.SHIFT, ASWS.LEFTARC, ASWS.SHIFT, ASWS.LEFTARC, ASWS.SHIFT,
                      ASWS.RIGHTARC, ASWS.RIGHTARC, ASWS.RIGHTARC, ASWS.SHIFT, ASWS.RIGHTARC,
                      ASWS.RIGHTARC ]

hearingCorrect = [ ASWS.SHIFT, ASWS.SHIFT, ASWS.LEFTARC, ASWS.SHIFT, ASWS.SHIFT, ASWS.SHIFT, ASWS.SWAP, ASWS.SWAP, ASWS.SHIFT, ASWS.SHIFT, ASWS.SHIFT, ASWS.SWAP, ASWS.SWAP, 
                  ASWS.SHIFT, ASWS.SHIFT, ASWS.SHIFT, ASWS.SWAP, ASWS.SWAP, ASWS.LEFTARC, ASWS.RIGHTARC, ASWS.RIGHTARC, ASWS.SHIFT, ASWS.LEFTARC, ASWS.SHIFT, ASWS.SHIFT, ASWS.RIGHTARC,
                  ASWS.RIGHTARC, ASWS.SHIFT, ASWS.RIGHTARC, ASWS.RIGHTARC]

afterCorrect = [ ASWS.SHIFT, ASWS.SHIFT, ASWS.LEFTARC, ASWS.SHIFT, ASWS.SHIFT, ASWS.SWAP, ASWS.SHIFT, ASWS.SHIFT, ASWS.SWAP, ASWS.LEFTARC, ASWS.SHIFT, ASWS.SHIFT, ASWS.LEFTARC, ASWS.RIGHTARC, ASWS.RIGHTARC, ASWS.RIGHTARC ]
 
class TestASWithSwapStaticOracle(unittest.TestCase):
     
    def testOneWordExample(self):
        """Example: oneWordExample"""
      
        system = ASWS()
        soracle = ArcStandardWithSwapEagerOracle(system)
        transitions = oracle.buildStaticCorrectTransitions(oneWordExample[1], system, soracle)
        self.assertEqual(oneWordCorrect, transitions)
   
    def testMsCollinsExample(self):
        """Example: msCollinsExample"""
       
        system = ASWS()
        soracle = ArcStandardWithSwapEagerOracle(system)
        transitions = oracle.buildStaticCorrectTransitions(msCollinsExample[1], system, soracle)
        self.assertEqual(msCollinsCorrect, transitions)
          
 
    def testHearingExample(self):
        """Example: hearingExample"""
      
        system = ASWS()
        soracle = ArcStandardWithSwapEagerOracle(system)
        transitions = oracle.buildStaticCorrectTransitions(hearingExample[1], system, soracle)
        self.assertEqual(hearingCorrect, transitions)
 
    def testAfterExample(self):
        """Example: afterExample"""
      
        system = ASWS()
        soracle = ArcStandardWithSwapEagerOracle(system)
        transitions = oracle.buildStaticCorrectTransitions(afterExample[1], system, soracle)
        self.assertEqual(afterCorrect, transitions)
         

class TestASWithSwapLazyStaticOracle(unittest.TestCase):
    
    def testAfterExample(self):
        """Example: afterExample"""
      
        afterLazyCorrect = [ ASWS.SHIFT, ASWS.SHIFT, ASWS.LEFTARC, ASWS.SHIFT, ASWS.SHIFT, ASWS.SHIFT, ASWS.LEFTARC, ASWS.SWAP, ASWS.SHIFT, ASWS.SHIFT, ASWS.LEFTARC, ASWS.RIGHTARC, ASWS.RIGHTARC, ASWS.RIGHTARC ]
        
        system = ASWS()
        soracle = ArcStandardWithSwapLazyOracle(system)
        transitions = oracle.buildStaticCorrectTransitions(afterExample[1], system, soracle)
        self.assertEqual(afterLazyCorrect, transitions)

    def testLetterExample(self):
        """Example: letterExample"""
      
        system = ASWS()
        lazyoracle = ArcStandardWithSwapLazyOracle(system)
        
        letterCorrect = [ASWS.SHIFT, ASWS.SHIFT, ASWS.SHIFT, ASWS.RIGHTARC, ASWS.SWAP, ASWS.SHIFT, ASWS.SHIFT, 
                         ASWS.SHIFT, ASWS.SHIFT, ASWS.LEFTARC, ASWS.RIGHTARC, ASWS.SWAP, ASWS.SHIFT, ASWS.SHIFT, 
                         ASWS.LEFTARC, ASWS.RIGHTARC, ASWS.RIGHTARC, ASWS.SHIFT, ASWS.RIGHTARC, ASWS.RIGHTARC]
        
        lazyTransitions = oracle.buildStaticCorrectTransitions(letterExample[1], system, lazyoracle)
        
        self.assertEqual(lazyTransitions, letterCorrect)
        
    def testMPC_letter(self):
        """basic mpc test for letterExample"""
         
        tree = letterExample[1]
        self.assertFalse(tree.isProjective())
         
        builder = MPCBuilder()
        mpcs = builder.buildMPCs(tree)
         
        self.assertEqual(mpcs[1], mpcs[2])
        self.assertEqual(mpcs[3], mpcs[4])
        self.assertEqual(mpcs[5], mpcs[4])
        self.assertEqual(mpcs[3], mpcs[5])
         
        self.assertNotEqual(mpcs[0], mpcs[1])
        self.assertNotEqual(mpcs[2], mpcs[3])
        self.assertNotEqual(mpcs[5], mpcs[6])
        self.assertNotEqual(mpcs[7], mpcs[6])
         
    def testMPC_2(self):
        """basic mpc test"""
         
        tree = datatypes.Tree([1, -1, 5, 4, 1, 4])
        self.assertFalse(tree.isProjective())
         
        builder = MPCBuilder()
        mpcs = builder.buildMPCs(tree)
        
        self.assertCountEqual({1: 1, 2: 2, 4: 4, 5: 5, 0: 1, 3: 4}, mpcs)
         
        
        
if __name__ == "__main__":
    unittest.main()
    
    
