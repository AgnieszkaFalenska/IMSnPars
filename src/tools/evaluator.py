'''
Created on 14.11.2017

@author: falensaa
'''

class LazyTreeEvaluator(object):
    def __init__(self):
        self.correctArcs = 0
        self.correctLblArcs = 0
        self.allArcs = 0
        
    def processTree(self, sent, tree):
        for pos, tok in enumerate(sent):
            self.allArcs += 1
            if tok.getHeadPos() == tree.getHead(pos):
                self.correctArcs += 1 
                
                if tok.dep == tree.getLabel(pos):
                    self.correctLblArcs += 1
                
    def calcUAS(self): 
        return 100 * float(self.correctArcs) / self.allArcs
        
    def calcLAS(self): 
        return 100 * float(self.correctLblArcs) / self.allArcs
    
    def calcAcc(self, accName):
        if accName == "LAS":
            return self.calcLAS()
        elif accName == "UAS":
            return self.calcUAS()
        else:
            raise Exception("Unknown accuracy: " + accName)  
    
    def reset(self):
        self.correctArcs = 0
        self.correctLblArcs = 0
        self.allArcs = 0
        