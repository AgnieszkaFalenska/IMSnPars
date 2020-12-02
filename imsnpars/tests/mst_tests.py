#!/usr/bin/env python
# coding=utf-8

'''
Created on 21.08.2017

@author: falensaa
'''

import unittest
from nparser.graph.mst.cle import ChuLiuEdmonds
import numpy as np

class TestSlowAndUglyCLEFullAlgorithm(unittest.TestCase):
  
    def testExample_oneCycle1(self):
        '''One cycle v.1'''
        
        weights =[ [ -np.inf, 9, 10, 9], [ -np.inf, -np.inf, 20, 3], [ -np.inf, 30, -np.inf, 30], [ -np.inf, 11, 0, -np.inf] ]
      
        cle = ChuLiuEdmonds()
          
        mst = cle.findMST(weights)
        self.assertCountEqual(mst.nodes(), [0, 1, 2, 3] )
        self.assertCountEqual(mst.edges(), [ (0, 2), (2, 1), (2, 3) ] )
          
         
    def testExample_oneCycle2(self):
        '''One cycle v.2'''
        
        weights = [ [ -np.inf, 3, 10, 5], [ -np.inf, -np.inf, 1, 10], [ -np.inf, 10, -np.inf, 8], [ -np.inf, 20, 5, -np.inf] ]
     
        cle = ChuLiuEdmonds()
         
        mst = cle.findMST(weights)
        self.assertCountEqual(mst.nodes(), [0, 1, 2, 3] )
        self.assertCountEqual(mst.edges(), [ (0, 2), (2, 3), (3, 1) ] )
         
    def testExample_twoCycles(self):
        '''Two cycles'''
        
        weights = [ [ -np.inf, 3, 2, 5], [ -np.inf, -np.inf, 4, 8], [ -np.inf, 4, -np.inf, 10], [ -np.inf, 8, 11, -np.inf] ]
     
        cle = ChuLiuEdmonds()
         
        mst = cle.findMST(weights)
        self.assertCountEqual(mst.nodes(), [0, 1, 2, 3] )
        self.assertCountEqual(mst.edges(), [ (0, 3), (3, 1), (3, 2) ] )
         
    def testExample_oneCycle3(self):
        '''One cycle v.3'''
        
        weights = [ [ -np.inf, 8, 20, 9], [ -np.inf, -np.inf, 10, 30], [ -np.inf, 3, -np.inf, 5], [ -np.inf, 12, 1, -np.inf] ]
     
        cle = ChuLiuEdmonds()
         
        mst = cle.findMST(weights)
        self.assertCountEqual(mst.nodes(), [0, 1, 2, 3] )
        self.assertCountEqual(mst.edges(), [ (0, 1), (0, 2), (1, 3) ] )
        
    def testExample_noCycle1(self):
        '''No cycle v.1'''
        
        weights = [ [ -np.inf, 6, 30, 10], [ -np.inf, -np.inf, 20, 20], [ -np.inf, 15, -np.inf, 12], [ -np.inf, 10, 25, -np.inf] ]
     
        cle = ChuLiuEdmonds()
         
        mst = cle.findMST(weights)
        self.assertCountEqual(mst.nodes(), [0, 1, 2, 3] )
        self.assertCountEqual(mst.edges(), [ (0, 2), (1, 3), (2, 1) ] )
        
    def testExample_noCycle2(self):
        '''No cycle v.2'''
        
        weights = [ [ -np.inf, 30, 26, 15], [ -np.inf, -np.inf, 23, 9], [ -np.inf, 25, -np.inf, 11], [ -np.inf, 10, 2, -np.inf] ]
     
        cle = ChuLiuEdmonds()
         
        mst = cle.findMST(weights)
        self.assertCountEqual(mst.nodes(), [0, 1, 2, 3] )
        self.assertCountEqual(mst.edges(), [ (0, 1), (0, 2), (0, 3) ] )
        
    def testExample_oneCycle4(self):
        '''One cycle v. 4'''
        
        weights = [ [ -np.inf, 9, 10, 9], [ -np.inf, -np.inf, 20, 3], [ -np.inf, 30, -np.inf, 30], [ -np.inf, 11, 0, -np.inf] ]
     
        cle = ChuLiuEdmonds()
         
        mst = cle.findMST(weights)
        self.assertCountEqual(mst.nodes(), [0, 1, 2, 3] )
        self.assertCountEqual(mst.edges(), [ (0, 2), (2, 1), (2, 3) ] )
        
    def testExample_oneCycle5(self):
        '''One cycle v. 5'''
        
        weights = [ [ -np.inf, 5, 10, 15], [ -np.inf, -np.inf, 25, 25], [ -np.inf, 20, -np.inf, 15], [ -np.inf, 10, 30, -np.inf] ]
     
        cle = ChuLiuEdmonds()
         
        mst = cle.findMST(weights)
        self.assertCountEqual(mst.nodes(), [0, 1, 2, 3] )
        self.assertCountEqual(mst.edges(), [ (0, 3), (2, 1), (3, 2) ] )
        
    def testExample_oneCycle6(self):
        '''One cycle v. 6'''
        
        weights = [ [ -np.inf, 15, 10, 5], [ -np.inf, -np.inf, 20, 5], [ -np.inf, 30, -np.inf, 10], [ -np.inf, 10, 5, -np.inf] ]
     
        cle = ChuLiuEdmonds()
         
        mst = cle.findMST(weights)
        self.assertCountEqual(mst.nodes(), [0, 1, 2, 3] )
        self.assertCountEqual(mst.edges(), [ (0, 2), (2, 1), (2, 3) ] )
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    







    
    
    
    
    
    
    
    