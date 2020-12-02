#!/usr/bin/env python
# coding=utf-8

'''
Created on 22.08.2017

@author: falensaa
'''

import unittest
from tools import datatypes
from nparser.trans.tsystem import oracle
from nparser.trans.tsystem.arcstandard import ArcStandard
from nparser.trans.tsystem.arcstandard import ArcStandardStaticOracle

class TestTreeMethods(unittest.TestCase):

    def testNonProjective_1(self):
        """short nonprojective tree with one nonprojective arc"""
        
        tree = datatypes.Tree([5, -1, 3, 1, 3, 1, 1])
        self.assertFalse(tree.isProjective())

    def testNonProjective_shortestLeft(self):
        """shortest nonprojective tree with left non projective arc"""
        
        tree = datatypes.Tree([-1, 3, 0, 0])
        self.assertFalse(tree.isProjective())
        
    def testNonProjective_shortestRight(self):
        """shortest nonprojective tree with right non projective arc"""
        
        tree = datatypes.Tree([-1, 0, 0, 1])
        self.assertFalse(tree.isProjective())
        
    def testNonProjective_2(self):
        """very long nonprojective tree"""
        
        tree = datatypes.Tree([3, 3, 3, 4, -1, 7, 7, 4, 4, 10, 4, 10, 11, 15, 15, 12, 15, 20, 20, 20, 16, 20, 25, 25, 25, 20, 4, 4, 27, 28, 31, 29, 31, 32, 35, 33, 35, 36, 37, 38, 45, 40, 43, 44, 35, 44, 45, 46, 49, 47, 46, 53, 53, 50, 53, 54, 55, 56, 59, 56, 59, 62, 60, 4])
        self.assertFalse(tree.isProjective())
    
    def testNonProjective_3(self):
        """long nonprojective tree"""
        
        tree = datatypes.Tree([3, 2, 14, 2, 5, 6, 3, 9, 9, 6, 14, 14, 13, 14, -1, 14, 14, 18, 14, 20, 18, 14])
        self.assertFalse(tree.isProjective())
        
    def testProjective_1(self):
        """one word tree"""
        
        tree = datatypes.Tree([ -1 ])
        self.assertTrue(tree.isProjective())

    def testProjective_2(self):
        """short projective tree v. 1"""
        
        tree = datatypes.Tree([ 1, -1, 1, 1, 5, 6, 3, 6, 1 ])
        self.assertTrue(tree.isProjective())
    
    def testProjective_3(self):
        """short projective tree v. 2"""
        
        tree = datatypes.Tree([1, 5, 1, 4, 1, -1, 7, 5, 5])
        self.assertTrue(tree.isProjective())
        
    def testHasArc_rootDep(self):
        tree = datatypes.Tree([-1, 0, 0, 1])
        self.assertFalse(tree.hasArc(1, -1))
        
    def testGetLabel_rootDep(self):
        tree = datatypes.Tree([-1, 0, 0, 1], [ "a", "b", "c", "d"])
        self.assertEqual(tree.getLabel(-1), None)
        
    def testGetHead_rootDep(self):
        tree = datatypes.Tree([-1, 0, 0, 1], [ "a", "b", "c", "d"])
        self.assertEqual(tree.getHead(-1), None)
        
if __name__ == "__main__":
    unittest.main()