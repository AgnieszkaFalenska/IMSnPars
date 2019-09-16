'''
Created on 21.09.2018

@author: falensaa
'''

import abc
import numpy as np

from tools import datatypes

class MaximumSpanningTreeAlgorithm(object):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self):
        pass
    
    def findMSTTree(self, scores):
        heads = self.findMSTHeads(scores)
        return self.__buildTree(heads)
    
    @abc.abstractmethod
    def findMSTHeads(self, scores):
        return

    @abc.abstractmethod
    def handlesNonProjectiveTrees(self):
        return
    
    def __buildTree(self, heads):
        predictTree = datatypes.Tree([ el - 1 for el in heads[1:] ])
        return predictTree
    
class SquareArrayScores(object):
    def __init__(self, length):
        self.length = length
        self.outputs = [ None ] * ( length * length )
        self.scores = np.zeros(length * length)
        
    def addScore(self, hId, dId, score):
        self.scores[self.__get_index(hId, dId)] = score
        
    def addOutput(self, hId, dId, output):
        self.outputs[self.__get_index(hId, dId)] = output
        
    def getScore(self, hId, dId):
        return self.scores[self.__get_index(hId, dId)]
    
    def getOutput(self, hId, dId):
        return self.outputs[self.__get_index(hId, dId)]
    
    def __get_index(self, hId, dId):
        index = hId + 1
        index = index * self.length + dId + 1
        return index
    
class CubicArrayScores(object):
    
    def __init__(self, length):
        self.length = length
        self.outputs = [ None ] * ( length * length * length )
        self.scores = np.zeros(length * length * length)
        
    def addScore(self, id1, id2, id3, score):
        self.scores[self.__get_index(id1, id2, id3)] = score
        
    def addOutput(self, id1, id2, id3, output):
        self.outputs[self.__get_index(id1, id2, id3)] = output
        
    def getScore(self, id1, id2, id3):
        return self.scores[self.__get_index(id1, id2, id3)]
    
    def getOutput(self, id1, id2, id3):
        return self.outputs[self.__get_index(id1, id2, id3)]
    
    def __get_index(self, id1, id2, id3):
        index = id1 + 1
        index = index * self.length + id2 + 1
        index = index * self.length + id3 + 1
        return index
    