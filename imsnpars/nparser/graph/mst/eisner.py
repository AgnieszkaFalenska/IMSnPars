'''
Created on Sep 12, 2019

@author: falensaa
'''

from nparser.graph.mst.algorithms import cyEisnerO2g, cyEisnerO2sib, cyEisner
from nparser.graph.mst.gdatatypes import MaximumSpanningTreeAlgorithm, SquareArrayScores, CubicArrayScores

class Eisner(MaximumSpanningTreeAlgorithm):
    def __init__(self):
        pass
    
    def emptyScores(self, instance):
        return SquareArrayScores(len(instance.sentence) + 1)
    
    def handlesNonProjectiveTrees(self):
        return False
    
    def findMSTHeads(self, scores):
        return cyEisner.decodeProjectiveE(scores.length, scores.scores)
    
class EisnerO2sib(MaximumSpanningTreeAlgorithm):
    def __init__(self):
        pass
    
    def emptyScores(self, instance):
        return CubicArrayScores(len(instance.sentence) + 1)
    
    def handlesNonProjectiveTrees(self):
        return False
    
    def findMSTHeads(self, scores):
        return cyEisnerO2sib.decodeProjective(scores.length, scores.scores)
    
class EisnerO2g(MaximumSpanningTreeAlgorithm):
    def __init__(self):
        pass
    
    def emptyScores(self, instance):
        return CubicArrayScores(len(instance.sentence) + 1)
    
    def handlesNonProjectiveTrees(self):
        return False
    
    def findMSTHeads(self, scores):
        return cyEisnerO2g.decodeProjective(scores.length, scores.scores)
    