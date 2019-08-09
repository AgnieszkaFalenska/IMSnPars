'''
Created on 13.11.2017

@author: falensaa
'''

import abc

class TrainInstance(object):
    def __init__(self, sentence, instanceIds):
        self.sentence = sentence
        self.ids = instanceIds
    
class VectorManager(object):
    def __init__(self, wordVectors, rootVector):
        self.wordsV = wordVectors
        self.rootV = rootVector
        
        
class NNetwork(abc.ABC):
    
    @abc.abstractmethod
    def initializeParameters(self, model, inputDim, outDim):
        return
        
    @abc.abstractmethod
    def renewNetwork(self):
        return
       
    @abc.abstractmethod
    def buildOutput(self, inputRepr, isTraining):
        return
    
    @abc.abstractmethod
    def buildLoss(self, output, correct, predicted):
        return
    
class NNTreeTask(abc.ABC):
    
    @abc.abstractmethod
    def initializeParameters(self, model, reprDim):
        pass
             
    @abc.abstractmethod
    def renewNetwork(self):
        pass
        
    @abc.abstractmethod
    def buildLosses(self, vectors, instance, correctTree, currectEpoch, predictTrain):
        pass
    
    @abc.abstractmethod
    def predict(self, instance, vectors):
        pass