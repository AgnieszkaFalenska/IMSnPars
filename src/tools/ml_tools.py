'''
Created on Mar 22, 2018

@author: falensaa
'''

import abc
import logging
import datetime

class TrainingManager:
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def prepareTrainingSplits(self, trainData, devData):
        return
    
    @abc.abstractmethod
    def startTraining(self):
        return
    
    @abc.abstractmethod
    def finishTraining(self):
        pass
    
    @abc.abstractmethod
    def isEndOfTraining(self):
        return
    
    @abc.abstractmethod
    def nextEpochNoAcc(self):
        pass
    
    @abc.abstractmethod
    def nextEpoch(self, accName, acc):
        pass
    
class AllEpochsTrainingManager(TrainingManager):
    def __init__(self, nrOfEpochs = 10, saveFun = None):
        self.__nrOfEpochs = nrOfEpochs
        self.__saveFun = saveFun
        
        # current epoch
        self.__currentEpoch = None
        self.__startTime = None
        
        self.__logger = logging.getLogger(self.__class__.__name__)
    
    def prepareTrainingSplits(self, trainData, devData):
        return trainData, devData
     
    def startTraining(self):
        self.__currentEpoch = 0
        self.__startTime = datetime.datetime.now()
        
    def finishTraining(self):
        self.__saveFun()
    
    def isEndOfTraining(self):
        return self.__currentEpoch >= self.__nrOfEpochs
     
    def nextEpochNoAcc(self):
        self.__logger.info("Epoch %i, Train time: %f" % (self.__currentEpoch, (datetime.datetime.now() - self.__startTime).total_seconds()))
        self.__currentEpoch += 1
        self.__startTime = datetime.datetime.now()
        
    def nextEpoch(self, acc):
        self.__logger.info("Epoch %i, Train time: %f" % (self.__currentEpoch, (datetime.datetime.now() - self.__startTime).total_seconds()))
        self.__currentEpoch += 1
        self.__startTime = datetime.datetime.now()
        
    def getCurrectEpoch(self):
        return self.__currentEpoch
        
class SaveMaxTrainingManager(TrainingManager):
    def __init__(self, nrOfEpochs, saveLast, saveFun, doneEpochs = 0, maxSoFar = -1):
        self.__nrOfEpochs = nrOfEpochs
        self.__saveFun = saveFun
        self.__saveLast = saveLast
        
        # current epoch
        self.__currentEpoch = None
        self.__startTime = None
        self.__bestAcc = None
        
        self.__doneEpochs = doneEpochs
        self.__maxSoFar = maxSoFar
        
        self.__logger = logging.getLogger(self.__class__.__name__)
    
    def prepareTrainingSplits(self, trainData, devData):
        return trainData, devData
     
    def startTraining(self):
        self.__currentEpoch = self.__doneEpochs
        self.__bestAcc = self.__maxSoFar
        self.__startTime = datetime.datetime.now()
        self.__logger.info("Started training for %i epochs (already done %i)" % (self.__nrOfEpochs, self.__doneEpochs))

    def finishTraining(self):
        self.__saveLast()
        return
    
    def isEndOfTraining(self):
        return self.__currentEpoch >= self.__nrOfEpochs
     
    def nextEpochNoAcc(self):
        self.__logger.warning("Using SaveMaxTrainingManager without dev accuracy does not make much sense")
        self.__logger.info("Epoch %i, Train time: %f" % (self.__currentEpoch, (datetime.datetime.now() - self.__startTime).total_seconds()))
        self.__currentEpoch += 1
        self.__startTime = datetime.datetime.now()
        
    def nextEpoch(self, acc):
        self.__logger.info("Epoch %i, Train time: %f" % (self.__currentEpoch, (datetime.datetime.now() - self.__startTime).total_seconds()))
        
        if acc > self.__bestAcc:
            self.__logger.info("Epoch %i, new better accuracy %f (was %f)" % (self.__currentEpoch, acc, self.__bestAcc))
            self.__bestAcc = acc
            
            # saving model
            self.__saveFun()
        
        self.__currentEpoch += 1
        self.__startTime = datetime.datetime.now()
        
    def getCurrectEpoch(self):
        return self.__currentEpoch

    def resetTraining(self, updateSecond):
        self.__logger.info("Reset training %s" % str(updateSecond))
        self.__currentEpoch = None
        self.__startTime = None
        self.__bestAcc = None

class EarlyStopTrainingManager(TrainingManager):
    def __init__(self, nrOfEpochs = 30, saveFun = None, patience = 2, splitTrain = 0.05):
        self.__nrOfEpochs = nrOfEpochs
        self.__saveFun = saveFun
        
        # current epoch
        self.__currentEpoch = None
        self.__startTime = None
        self.__lastAcc = None
        self.__worseEpochs = None
        
        # early stopping options
        self.__patience = patience
        self.__splitTrain = splitTrain
        
        self.__logger = logging.getLogger(self.__class__.__name__)
    
    def prepareTrainingSplits(self, trainData, devData):
        if devData == None:
            partSize = int(self.__splitTrain * len(trainData))
            devData = trainData[partSize:]
            trainData = trainData[:partSize]
            self.__logger.info("Splitting training data for early stop: %i, %i" % (len(trainData), len(devData)))
            
        return trainData, devData
     
    def startTraining(self):
        self.__currentEpoch = 0
        self.__worseEpochs = 0
        self.__lastAcc = -1
        self.__startTime = datetime.datetime.now()
        self.__logger.info("Started training for %i epochs and patience %i" % (self.__nrOfEpochs, self.__patience))

    def finishTraining(self):
        # the model has been already saved
        return
    
    def isEndOfTraining(self):
        return self.__currentEpoch >= self.__nrOfEpochs
     
    def nextEpochNoAcc(self):
        self.__logger.warning("Using EarlyStopTrainingManager without dev accuracy does not make much sense")
        self.__logger.info("Epoch %i, Train time: %f" % (self.__currentEpoch, (datetime.datetime.now() - self.__startTime).total_seconds()))
        self.__currentEpoch += 1
        self.__startTime = datetime.datetime.now()
        
    def nextEpoch(self, acc):
        self.__logger.info("Epoch %i, Train time: %f" % (self.__currentEpoch, (datetime.datetime.now() - self.__startTime).total_seconds()))
        
        if acc <= self.__lastAcc:
            self.__worseEpochs += 1
            self.__logger.info("Epoch %i, accuracy worse than the last (was %f)" % (self.__currentEpoch, self.__lastAcc))
            
            if self.__worseEpochs > self.__patience:
                self.__logger.info("Epoch %i, finishing training with %i worse epochs" % (self.__currentEpoch, self.__worseEpochs))
                self.__currentEpoch = self.__nrOfEpochs
                return
        else:
            self.__logger.info("Epoch %i, new better accuracy %f (was %f)" % (self.__currentEpoch, acc, self.__lastAcc))
            self.__worseEpochs = 0
            self.__lastAcc = acc
            
            # saving model
            self.__saveFun()
        
        self.__currentEpoch += 1
        self.__startTime = datetime.datetime.now()
        
    def getCurrectEpoch(self):
        return self.__currentEpoch
