'''
Created on 23.08.2017

@author: falensaa
'''

import dynet
import logging
import random
import datetime

from tools import datatypes, utils, evaluator
       
class NNParserTrainLogger(object):
    """Class accumulates all the logging, tracking, and printing of the parser"""
    
    def __init__(self, reportTrain):
        
        if reportTrain:
            self.trainEval = evaluator.LazyTreeEvaluator()
        else:
            self.trainEval = None
        
        self.epochNr = 0
        self.__reset()
        self.__logger = logging.getLogger(self.__class__.__name__)
        
    def finishEpoch(self, uas = None, las = None):
        if uas is not None and las is not None:
            self.__logger.debug("Epoch %i, UAS=%.2f LAS=%.2f" % ( self.epochNr, uas, las))
        
        if self.trainEval:
            trainLAS = self.trainEval.calcLAS()
            trainUAS = self.trainEval.calcUAS()
            self.__logger.debug("Epoch %i, Train UAS=%.2f Train LAS=%.2f" % ( self.epochNr, trainUAS, trainLAS))
        
        self.__reset()
        self.epochNr += 1
        
    def finishInstance(self):
        self.seenInstances += 1
        
    def finishUpdate(self, loss, nrOfUpdates):
        self.iLoss += loss
        self.iNrOfUpdates += nrOfUpdates 
          
    def finishSentence(self, sentence, predictedTree, lossesAfterUpdate):
        if self.trainEval:
            self.trainEval.processTree(sentence, predictedTree)
         
        self.seenSentences += 1
        if (self.seenSentences > 1 and self.seenSentences % 100 == 0):
            thisBatchLoss = 0 if len(lossesAfterUpdate) == 0 else dynet.esum(lossesAfterUpdate).value()
            currLoss = self.iLoss + thisBatchLoss
            currNrOfUpdates = self.iNrOfUpdates + len(lossesAfterUpdate)
            time = (datetime.datetime.now() - self.startTime).total_seconds()
            self.__logger.debug("Epoch %i, Sent %i, AVG loss %.2f, AVG updates %.2f, Time %.2gs" % ( self.epochNr,
                                                                                                 self.seenSentences, 
                                                                                                 currLoss / self.seenInstances,
                                                                                                 float(currNrOfUpdates) / self.seenInstances,
                                                                                                 time ))
                    
            self.iLoss = -thisBatchLoss
            self.iNrOfUpdates = -len(lossesAfterUpdate)
            self.seenInstances = 0
            self.startTime = datetime.datetime.now()
             
    def __reset(self):
        self.seenSentences = 0
        self.seenInstances = 0
        self.iNrOfUpdates = 0
        self.iLoss = 0
        self.startTime = datetime.datetime.now()
        
        if self.trainEval:
            self.trainEval.reset()
        

class NDependencyParser(object):
    def __init__(self, reprBuilder, parsingTask, lblTask, trainer, lossBatchSize=0):
        self.__logger =  logging.getLogger(self.__class__.__name__)
        
        self.__model = None
        
        # representations
        self.__reprBuilder = reprBuilder
        self.__trainer = trainer
        self.__lossBatchSize = lossBatchSize
         
        self.__parser = parsingTask
        self.__labeler = lblTask
        self.__logger = logging.getLogger(self.__class__.__name__)
    
    def save(self, filename):
        with open(filename, 'wb') as pickleOut:
            # saving all w2i dictionaries
            self.__reprBuilder.save(pickleOut)
        
            # saving labels
            self.__labeler.save(pickleOut)
        
        # saving parameters
        self.__model.save(filename + ".params")
    
    def load(self, filename):
        with open(filename, 'rb') as pickleIn:
            # loading all w2i dictionaries
            self.__reprBuilder.load(pickleIn)
        
            # loading labels
            self.__labeler.load(pickleIn)
        
        # loading parameters
        self.__model = dynet.ParameterCollection()
        self.__initializeParameters()
        self.__model.populate(filename + ".params")
    
    def __initializeParameters(self):
        self.__reprBuilder.initializeParameters(self.__model)
        self.__parser.initializeParameters(self.__model, self.__reprBuilder.getDim())
        self.__labeler.initializeParameters(self.__model, self.__reprBuilder.getDim())
    
    def getReprBuilder(self):
        return self.__reprBuilder
    
    def getParsingTask(self):
        return self.__parser
    
    def train(self, allSentences, trainManager, devData, predictTrain):
        if not self.__parser.handlesNonProjectiveTrees():    
            sentences = utils.filterNonProjective(allSentences)
            self.__logger.info("Filtered %i non-projective trees " % (len(allSentences) - len(sentences)))
        else:
            sentences = allSentences
        
        # fill all dictionaries
        self.__reprBuilder.readData(sentences)
        self.__labeler.readData(sentences)
        
        # initialize parameters
        self.__model = dynet.ParameterCollection()
        self.__initializeParameters()
        
        # for logging
        trainLogger = NNParserTrainLogger(predictTrain)
        self.__trainOnSentences(sentences, devData, trainManager, trainLogger, predictTrain)
    
    
    def predict(self, sentences, writer):
        startTime = datetime.datetime.now()
        for sent in sentences:
            self.__renewNetwork()
            instance = self.__reprBuilder.buildInstance(sent)
            predictedTree = self.__predictTree(instance)
            writer.processTree(sent, predictedTree)
            
        endTime = datetime.datetime.now()
        self.__logger.info("Predict time: %f" % ((endTime - startTime).total_seconds()))
    
    def __trainOnSentences(self, sentences, devData, trainManager, trainLogger, predictTrain):
        # start training
        trainer = self.__trainer(self.__model)
            
        trainManager.startTraining()
        while not trainManager.isEndOfTraining():
            random.shuffle(sentences)
            
            # build new instances -- all the dropouts will re-run
            instances = [ self.__reprBuilder.buildInstance(sent) for sent in sentences ]
     
            # set the correct trees
            for instance in instances:
                instance.correctTree = datatypes.sentence2Tree(instance.sentence)
                

            self.__renewNetwork()
            
            # one batch of losses
            losses = [ ]
            
            for instance in instances:
                vectors = self.__reprBuilder.prepareVectors(instance, isTraining=True)
                
                parsLosses, predictTree = self.__parser.buildLosses(vectors, instance, currentEpoch = trainManager.getCurrectEpoch(), predictTrain = predictTrain)
                lblLosses, predictLbls = self.__labeler.buildLosses(vectors, instance, currentEpoch = trainManager.getCurrectEpoch(), predictTrain = predictTrain)
                
                if predictTrain and predictLbls:
                    predictTree.setLabels(predictLbls)
            
                trainLogger.finishInstance()
                
                losses.extend(parsLosses + lblLosses)

                # update
                if len(losses) > self.__lossBatchSize:
                    loss = dynet.esum(losses)
                    lossValue = loss.scalar_value() # equivalent to loss.forward()
                    trainLogger.finishUpdate(lossValue, len(losses))
                    
                    loss.backward()
                    trainer.update()
                        
                    losses = [ ]
                    self.__renewNetwork()
        
                trainLogger.finishSentence(instance.sentence, predictTree, losses)
            
            if len(losses) > 0:        
                loss = dynet.esum(losses)
                loss.forward()
                loss.backward()
                trainer.update()
                   
            if devData != None:
                lazyEval = evaluator.LazyTreeEvaluator()
                self.predict(devData, lazyEval)
                
                trainLogger.finishEpoch(lazyEval.calcUAS(), lazyEval.calcLAS())
                
                if self.__labeler.reportLabels():
                    trainManager.nextEpoch(lazyEval.calcLAS()) 
                else:
                    trainManager.nextEpoch(lazyEval.calcUAS())
            else:
                trainManager.nextEpochNoAcc()
                trainLogger.finishEpoch()
            
        trainManager.finishTraining()
        
    def __predictTree(self, instance):
        vectors = self.__reprBuilder.prepareVectors(instance, isTraining=False)
        predictTree = self.__parser.predict(instance, vectors)
        lbls = self.__labeler.predict(instance, predictTree, vectors)
        
        if lbls != None:
            predictTree.setLabels(lbls)
            
        return predictTree
        
    def __renewNetwork(self):
        dynet.renew_cg()
        self.__parser.renewNetwork()
        self.__labeler.renewNetwork()
        
