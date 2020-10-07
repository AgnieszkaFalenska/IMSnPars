'''
Created on 23.08.2017

@author: falensaa
'''

import dynet
import logging
import random
import datetime

from nparser.task import NNParserTrainLogger
from tools import evaluator, datatypes

class DNDependencyMultitaskParser(object):
    def __init__(self, reprBuilder, firstTask, secondTask, lblTask, secondLblTask, trainer, lossBatchSize=0):
        self.__logger =  logging.getLogger(self.__class__.__name__)
        
        self.__firstTask = firstTask
        self.__secondTask = secondTask
        self.__labeler = lblTask
        self.__secondLabeler = secondLblTask
        self.__trainer = trainer
        self.__lossBatchSize = lossBatchSize
        
        ###
        # common
        self.__model = None
        self.__reprBuilder = reprBuilder
    
    def getParsingTask(self):
        return self.__firstTask
    
    def getReprBuilder(self):
        return self.__reprBuilder
    
    def save(self, filename):
        with open(filename, 'wb') as pickleOut:
            # saving all w2i dictionaries
            self.__reprBuilder.save(pickleOut)
            self.__labeler.save(pickleOut)
            self.__secondLabeler.save(pickleOut)
        
        # saving parameters
        self.__model.save(filename + ".params")
    
    def load(self, filename):
        with open(filename, 'rb') as pickleIn:
            # loading all w2i dictionaries
            self.__reprBuilder.load(pickleIn)
            self.__labeler.load(pickleIn)
            self.__secondLabeler.load(pickleIn)
        
        # loading parameters
        self.__model = dynet.ParameterCollection()
        self.__initializeParameters()
        self.__model.populate(filename + ".params")
    
    def __initializeParameters(self):
        self.__reprBuilder.initializeParameters(self.__model)
        self.__firstTask.initializeParameters(self.__model, self.__reprBuilder.getDim())
        self.__secondTask.initializeParameters(self.__model, self.__reprBuilder.getDim())
        self.__labeler.initializeParameters(self.__model, self.__reprBuilder.getDim())
        self.__secondLabeler.initializeParameters(self.__model, self.__reprBuilder.getDim())
        
    def train(self, allSentences, trainManager, devData, predictTrain):
        sentences = allSentences
        
        # fill all dictionaries
        self.__reprBuilder.readData(sentences)
        self.__labeler.readData(sentences)
        self.__secondLabeler.readData(sentences)
        
        # initialize parameters
        self.__model = dynet.ParameterCollection()
        self.__initializeParameters()
        
        # for logging
        trainLogger = NNParserTrainLogger(predictTrain)
        
        # start training
        trainer = self.__trainer(self.__model)
        
        checkProj = not self.__firstTask.handlesNonProjectiveTrees() or not self.__secondTask.handlesNonProjectiveTrees()
        
        trainManager.startTraining()
        while not trainManager.isEndOfTraining():
            random.shuffle(sentences)
            instances = [ self.__reprBuilder.buildInstance(sent) for sent in sentences ]
            
            # set the correct trees
            for instance in instances:
                instance.correctTree = datatypes.sentence2Tree(instance.sentence)
                
                if checkProj:
                    instance.isProjTree = instance.correctTree.isProjective()

            self.__renewNetwork()
            
            # one batch
            losses = [ ]
            for instance in instances:
                vectors = self.__reprBuilder.prepareVectors(instance, isTraining=True)
                
                if not self.__firstTask.handlesNonProjectiveTrees() and not instance.isProjTree:
                    firstTaskLosses, firstTaskPredictTree = [ ], None
                else:
                    firstTaskLosses, firstTaskPredictTree = self.__firstTask.buildLosses(vectors, instance, currentEpoch = trainManager.getCurrectEpoch(), predictTrain = predictTrain)
                    
                if not self.__secondTask.handlesNonProjectiveTrees() and not instance.isProjTree:
                    secondTaskLosses, secondTaskPredictTree = [ ], None
                else:
                    secondTaskLosses, secondTaskPredictTree = self.__secondTask.buildLosses(vectors, instance, currentEpoch = trainManager.getCurrectEpoch(), predictTrain = predictTrain)
            
                firstLblLosses, firstPredictLbls = self.__labeler.buildLosses(vectors, instance, currentEpoch = trainManager.getCurrectEpoch(), predictTrain = predictTrain)
                secondLblLosses, secondPredictLbls = self.__secondLabeler.buildLosses(vectors, instance, currentEpoch = trainManager.getCurrectEpoch(), predictTrain = predictTrain)
                
                trainLogger.finishInstance()
                
                if predictTrain and firstPredictLbls and not firstTaskPredictTree.getLabels():
                    firstTaskPredictTree.setLabels(firstPredictLbls)
                
                if predictTrain and secondPredictLbls and not secondTaskPredictTree.getLabels():
                    secondTaskPredictTree.setLabels(secondPredictLbls)
                    
                ##
                # making update    
                
                ###
                # joining losses
                losses.extend(firstTaskLosses + secondTaskLosses + firstLblLosses + secondLblLosses)
                
                # update
                if len(losses) > self.__lossBatchSize:
                    random.shuffle(losses)
                    
                    loss = dynet.esum(losses)
                    trainLogger.finishUpdate(loss.value(), len(losses))
                    
                    loss.forward()
                    loss.backward()
                    trainer.update()
                        
                    losses = [ ]
                    self.__renewNetwork()
                
                trainLogger.finishSentence(instance.sentence, firstTaskPredictTree, losses)
                
            if len(losses) > 0:        
                loss = dynet.esum(losses)
                loss.forward()
                loss.backward()
                trainer.update()
                
            if devData != None:
                lazyEval = evaluator.LazyTreeEvaluator()
                lazyEvalSecondTask = evaluator.LazyTreeEvaluator()
                self.predict(devData, lazyEval, lazyEvalSecondTask)
                
                trainLogger.finishEpoch(lazyEval.calcUAS(), lazyEval.calcLAS())
                self.__logger.info("T2 UAS=%.2f T2 LAS=%.2f" % (lazyEvalSecondTask.calcUAS(), lazyEvalSecondTask.calcLAS()))
                
                if self.__labeler.reportLabels():
                    trainManager.nextEpoch(lazyEval.calcLAS()) 
                else:
                    trainManager.nextEpoch(lazyEval.calcUAS())
            else:
                trainManager.nextEpochNoAcc()
                trainLogger.finishEpoch()
            
        trainManager.finishTraining()
    
    def predict(self, sentences, writer, secondTaskWriter = None):
        startTime = datetime.datetime.now()
        
        for sent in sentences:
            self.__renewNetwork()
            
            instance = self.__reprBuilder.buildInstance(sent)
            vectors = self.__reprBuilder.prepareVectors(instance, isTraining=False)
            
            firstTaskPredictTree = self.__firstTask.predict(instance, vectors)
            if not firstTaskPredictTree.getLabels():
                t1Lbls = self.__labeler.predict(instance, firstTaskPredictTree, vectors)
                if t1Lbls:
                    firstTaskPredictTree.setLabels(t1Lbls)
                    
            writer.processTree(sent, firstTaskPredictTree)
            
            if secondTaskWriter is not None:
                secondTaskPredictTree = self.__secondTask.predict(instance, vectors)
                
                if not secondTaskPredictTree.getLabels() or secondTaskPredictTree.getLabels()[0] is None:
                    t2Lbls = self.__secondLabeler.predict(instance, secondTaskPredictTree, vectors)
                
                    if t2Lbls:
                        secondTaskPredictTree.setLabels(t2Lbls)
                
                secondTaskWriter.processTree(sent, secondTaskPredictTree)
            
            
        endTime = datetime.datetime.now()
        self.__logger.info("Predict time: %f" % ((endTime - startTime).total_seconds()))
        
    def __renewNetwork(self):
        dynet.renew_cg()
        self.__firstTask.renewNetwork()
        self.__secondTask.renewNetwork()
        self.__labeler.renewNetwork()
        self.__secondLabeler.renewNetwork()
    
