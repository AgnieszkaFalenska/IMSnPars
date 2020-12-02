'''
Created on 18.08.2017

@author: falensaa
'''

#import projpath

import unittest
from nparser.trans.tsystem import tdatatypes

class TestTreeUnderConstructionMethods(unittest.TestCase):
    
    def testEmptyTree(self):
        """at the beginning there is no head"""
        
        clear = tdatatypes.TreeUnderConstruction(5)
        self.assertFalse(clear.hasHead(0))
  
  
    def testAddArc(self): 
        """added arc exists in the collection"""
    
        coll = tdatatypes.TreeUnderConstruction(5)
        coll.add(0, 3)
        self.assertTrue(coll.hasHead(3))
  
    def testArcDirection(self):
        """added arc is in good direction"""
        
        coll = tdatatypes.TreeUnderConstruction(5)
        coll.add(0, 3)
        self.assertFalse(coll.hasHead(0))
  
    def testGetNrOfArcsIsEmpty(self):
        """nrOfArcs for no arc is zero"""
    
        coll = tdatatypes.TreeUnderConstruction(5)
        self.assertEqual(coll.getNrOfArcs(), 0)
  
  
    def testGetNrOfArcsNotEmpty(self):
        """nrOfArcs after adding two arcs is two"""
    
        coll = tdatatypes.TreeUnderConstruction(5)
        coll.add(0, 1)
        coll.add(0, 2)
        self.assertEqual(coll.getNrOfArcs(), 2)
  
  
    def testHasArcForNoArc(self):
        """hasArc for non existing arc is false"""
        
        coll = tdatatypes.TreeUnderConstruction(5)
        self.assertFalse(coll.hasArc(0, 1))
      
      
    def testHasArcForArc(self):
        """hasArc for existing arc is true"""
        
        coll = tdatatypes.TreeUnderConstruction(5)
        coll.add(0, 1)
        self.assertTrue(coll.hasArc(0, 1))
      
    def testHowManyChildrenNoArc(self):
        """when no arc - no children"""
        coll = tdatatypes.TreeUnderConstruction(5)
        self.assertEqual(coll.howManyChildren(0), 0)
        self.assertEqual(coll.howManyChildren(-1), 0)
    
    def testHowManyChildrenWithOneArc(self):
        """when only one arc - one child"""
        coll = tdatatypes.TreeUnderConstruction(5)
        coll.add(0, 1)
        self.assertEqual(coll.howManyChildren(0), 1)
        self.assertEqual(coll.howManyChildren(1), 0)
    
    def testHowManyChildrenWithFewArcs(self):
        """when no arc - no children"""
        coll = tdatatypes.TreeUnderConstruction(5)
        coll.add(0, 1)
        coll.add(0, 2)
        coll.add(0, 3)
        self.assertEqual(coll.howManyChildren(0), 3)    
        self.assertEqual(coll.howManyChildren(1), 0)
            
    def testLeftMostChildForNoChildren(self):
        """if there is no children left most is None"""
        
        coll = tdatatypes.TreeUnderConstruction(5)
        self.assertEqual(coll.getLeftChild(0), None)
        
    def testLeftMostChildForOneChild_onLeft(self):
        """if there is one child it is the left most"""
        
        coll = tdatatypes.TreeUnderConstruction(5)
        coll.add(1, 0)
        self.assertEqual(coll.getLeftChild(1), 0)
        
    def testLeftMostChildForOneChild_onRight(self):
        """if there is one child it is the left most"""
        
        coll = tdatatypes.TreeUnderConstruction(5)
        coll.add(1, 2)
        self.assertEqual(coll.getLeftChild(1), 2)
        
    def testLeftMostChildForFewChildren_inOrder(self):
        """if there is few children the smallest index is the left most"""
        
        coll = tdatatypes.TreeUnderConstruction(5)
        coll.add(1, 0)
        coll.add(1, 2)
        coll.add(1, 3)
        self.assertEqual(coll.getLeftChild(1), 0)
        
    def testLeftMostChildForFewChildren_notInOrder(self):
        """if there is few children the smallest index is the left most"""
        
        coll = tdatatypes.TreeUnderConstruction(5)
        coll.add(1, 2)
        coll.add(1, 0)
        coll.add(1, 3)
        self.assertEqual(coll.getLeftChild(1), 0)
        
    def testRightMostChildForNoChildren(self):
        """if there is no children right most is None"""
        
        coll = tdatatypes.TreeUnderConstruction(5)
        self.assertEqual(coll.getRightChild(0), None)
        
    def testRightMostChildForOneChild_onLeft(self):
        """if there is one child it is the right most"""
        
        coll = tdatatypes.TreeUnderConstruction(5)
        coll.add(1, 0)
        self.assertEqual(coll.getRightChild(1), 0)
        
    def testRightMostChildForOneChild_onRight(self):
        """if there is one child it is the right most"""
        
        coll = tdatatypes.TreeUnderConstruction(5)
        coll.add(1, 2)
        self.assertEqual(coll.getRightChild(1), 2)
        
    def testRightMostChildForFewChildren_inOrder(self):
        """if there is few children the biggest index is the right most"""
        
        coll = tdatatypes.TreeUnderConstruction(5)
        coll.add(1, 0)
        coll.add(1, 2)
        coll.add(1, 3)
        self.assertEqual(coll.getRightChild(1), 3)
        
    def testRightMostChildForFewChildren_notInOrder(self):
        """if there is few children the biggest index is the right most"""
        
        coll = tdatatypes.TreeUnderConstruction(5)
        coll.add(1, 0)
        coll.add(1, 3)
        coll.add(1, 2)
        self.assertEqual(coll.getRightChild(1), 3)
        
    def testLeftSecondChildForNoChildren(self):
        """if there is no children left second most is None"""
        
        coll = tdatatypes.TreeUnderConstruction(5)
        self.assertEqual(coll.getLeftChild(0, 1), None)
        
    def testLeftSecondChildForOneChild(self):
        """if there is one child the left second is None"""
        
        coll = tdatatypes.TreeUnderConstruction(5)
        coll.add(1, 0)
        self.assertEqual(coll.getLeftChild(1, 1), None)
        
    def testLeftSecondChildForTwoChildren_inOrder(self):
        """if there is two children the second left is not the smallest one"""
        
        coll = tdatatypes.TreeUnderConstruction(5)
        coll.add(1, 2)
        coll.add(1, 3)
        self.assertEqual(coll.getLeftChild(1, 1), 3)
        
    def testLeftSecondChildForTwoChildren_notInOrder(self):
        """if there is two children the second left is not the smallest one"""
        
        coll = tdatatypes.TreeUnderConstruction(5)
        coll.add(1, 3)
        coll.add(1, 2)
        self.assertEqual(coll.getLeftChild(1, 1), 3)
        
    def testLeftSecondChildForFewChildren_inOrder(self):
        """if there is few children the second smallest index is the left second most"""
        
        coll = tdatatypes.TreeUnderConstruction(10)
        coll.add(1, 0)
        coll.add(1, 2)
        coll.add(1, 3)
        coll.add(1, 4)
        coll.add(1, 5)
        self.assertEqual(coll.getLeftChild(1, 1), 2)
        
    def testLeftSecondChildForFewChildren_notInOrder(self):
        """if there is few children the second smallest index is the left second most"""
        
        coll = tdatatypes.TreeUnderConstruction(10)
        coll.add(1, 3)
        coll.add(1, 0)
        coll.add(1, 4)
        coll.add(1, 5)
        coll.add(1, 2)
        self.assertEqual(coll.getLeftChild(1, 1), 2)
        
    def testRightSecondChildForNoChildren(self):
        """if there is no children right second most is None"""
        
        coll = tdatatypes.TreeUnderConstruction(5)
        self.assertEqual(coll.getRightChild(0, 1), None)
        
    def testRightSecondChildForOneChild(self):
        """if there is one child the right second is None"""
        
        coll = tdatatypes.TreeUnderConstruction(5)
        coll.add(1, 0)
        self.assertEqual(coll.getRightChild(1, 1), None)
        
    def testRightSecondChildForTwoChildren_inOrder(self):
        """if there is two children the second right is not the biggest one"""
        
        coll = tdatatypes.TreeUnderConstruction(5)
        coll.add(1, 2)
        coll.add(1, 3)
        self.assertEqual(coll.getRightChild(1, 1), 2)
        
    def testRightSecondChildForTwoChildren_notInOrder(self):
        """if there is two children the second right is not the biggest one"""
        
        coll = tdatatypes.TreeUnderConstruction(5)
        coll.add(1, 3)
        coll.add(1, 2)
        self.assertEqual(coll.getRightChild(1, 1), 2)
        
    def testRightSecondChildForFewChildren_inOrder(self):
        """if there is few children the second biggest index is the right second most"""
        
        coll = tdatatypes.TreeUnderConstruction(10)
        coll.add(1, 0)
        coll.add(1, 2)
        coll.add(1, 3)
        coll.add(1, 4)
        coll.add(1, 5)
        self.assertEqual(coll.getRightChild(1, 1), 4)
        
    def testRightSecondChildForFewChildren_notInOrder(self):
        """if there is few children the second biggest index is the right second most"""
        
        coll = tdatatypes.TreeUnderConstruction(10)
        coll.add(1, 3)
        coll.add(1, 0)
        coll.add(1, 4)
        coll.add(1, 5)
        coll.add(1, 2)
        self.assertEqual(coll.getRightChild(1, 1), 4)
        
class TestStateMethods(unittest.TestCase):

    def testStartConfigurationStack(self):
        """start configuration has only root on the stack"""
        
        state = tdatatypes.State(5)
        self.assertEqual(state.stack.length(), 1)
  
    def testStartConfigurationBuffer(self):
        """start configuration has all tokens in the buffer"""
        
        state = tdatatypes.State(5)
        self.assertEqual(state.buffer.length(), 5)
      
    def testStartConfigurationArcs(self):
        """start configuration has no arcs"""
        
        state = tdatatypes.State(5)
        self.assertEqual(state.arcs.getNrOfArcs(), 0)    

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()