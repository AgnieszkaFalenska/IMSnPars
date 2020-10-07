'''
Created on 14.11.2017

@author: falensaa
'''

class LazyTreeEvaluator(object):
    def __init__(self, evalFullLabels = False):
        self.correctArcs = 0
        self.correctLblArcs = 0
        self.allArcs = 0
        self.__evalFullLabels = evalFullLabels
        
    def processTree(self, sent, tree):
        for pos, tok in enumerate(sent):
            self.allArcs += 1
            if tok.getHeadPos() == tree.getHead(pos):
                self.correctArcs += 1 
               
                if self.__compareLabels(tok.dep, tree.getLabel(pos)):
                    self.correctLblArcs += 1
        
    def calcUAS(self): 
        return 100 * float(self.correctArcs) / self.allArcs
        
    def calcLAS(self): 
        return 100 * float(self.correctLblArcs) / self.allArcs
    
    def calcAcc(self, accName):
        if accName == "all":
            return (self.calcUAS(), self.calcLAS())
        
        if accName == "LAS":
            return self.calcLAS()
        
        if accName == "UAS":
            return self.calcUAS()
        
        raise Exception("Unknown accuracy: " + accName)  
    
    def reset(self):
        self.correctArcs = 0
        self.correctLblArcs = 0
        self.allArcs = 0
        
    def __compareLabels(self, dep1, dep2):
        if self.__evalFullLabels or dep1 is None or dep2 is None:
            return dep1 == dep2
        else:
            return dep1.split(":")[0] == dep2.split(":")[0]
