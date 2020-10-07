'''
Created on 23.08.2017

@author: falensaa
'''

import logging
import dynet
import sys

from nparser.task import NDependencyParser
from nparser.features import ZeroDummyVectorBuilder, RandomDummyVectorBuilder

import repr.builder as rbuilder
import nparser.graph.builder as gbuilder
import nparser.trans.builder as tbuilder
import nparser.labels.builder as lbuilder
import nparser.mtl.builder as mbuilder
import nparser.labels.task as ltask
import nparser.mtl.task as mtask


def buildParser(opts):
    reprBuilders = rbuilder.buildReprBuilders(opts)
    contextReprBuilder = rbuilder.buildContextReprBuilder(opts, reprBuilders)
    
    dummyFeatBuilder = __buildDummyFeatBuilder(opts, sum([ rd.getDim() for rd in reprBuilders ]), contextReprBuilder.getDim())
    trainer = __buildTrainer(opts)
    
    if opts.parser == "MTL":
        firstTask, secondTask = mbuilder.buildMTLParser(opts, dummyFeatBuilder, contextReprBuilder)
        
        firstLblTask = __buildLabeler(opts.t1Labeler, opts.firstTask, opts, dummyFeatBuilder, contextReprBuilder, firstTask)
        secondLblTask = __buildLabeler(opts.t2Labeler, opts.secondTask, opts, dummyFeatBuilder, contextReprBuilder, secondTask)
        
        return mtask.DNDependencyMultitaskParser(contextReprBuilder, firstTask, secondTask, firstLblTask, secondLblTask, trainer)
    
    if opts.parser == "TRANS":
        parsingTask = tbuilder.buildTransParser(opts, dummyFeatBuilder, contextReprBuilder)
        lblTask = __buildLabeler(opts.labeler, opts.parser, opts, dummyFeatBuilder, contextReprBuilder, parsingTask)
        
    elif opts.parser == "GRAPH":
        parsingTask = gbuilder.buildGraphParser(opts, dummyFeatBuilder, contextReprBuilder)
        lblTask = __buildLabeler(opts.labeler, opts.parser, opts, dummyFeatBuilder, contextReprBuilder, parsingTask)
        
    else:
        logging.error("Unknown parser: %s" % opts.parser)
        sys.exit()
    
    return NDependencyParser(contextReprBuilder, parsingTask, lblTask, trainer)
        
def __buildDummyFeatBuilder(opts, reprDim, outDim):
    if opts.dummy == "zero":
        return ZeroDummyVectorBuilder(outDim)
    elif opts.dummy == "random":
        return RandomDummyVectorBuilder(reprDim, outDim, opts.nonLinFun, separate = False)
    elif opts.dummy == "random_sep":
        return RandomDummyVectorBuilder(reprDim, outDim, opts.nonLinFun, separate = True)
    else:
        logging.error("Unknown dummy builder: %s" % opts.dummy)
        sys.exit()

def __buildTrainer(opts):
    if opts.trainer == "adam":
        trainerClass = dynet.AdamTrainer
    elif opts.trainer == "simple-sgd":
        trainerClass = dynet.SimpleSGDTrainer
    elif opts.trainer == "momentum-sgd":
        trainerClass = dynet.MomentumSGDTrainer
        
    if opts.learningRate is not None:
        if opts.trainer == "adam":
            trainer = lambda model : trainerClass(model, alpha=opts.learningRate)
        else:
            trainer = lambda model : trainerClass(model, learning_rate=opts.learningRate)
    else:
        trainer = trainer = lambda model : trainerClass(model)
          
    logging.info("Building trainer: %s with learning rate: %s" % ( trainerClass, "'default'" if not opts.learningRate else str(opts.learningRate)))  
    return trainer

def __buildLabeler(labeler, parser, opts, dummyBuilder, reprBuilder, parsingTask):
    if labeler == "graph-mtl":
        lblTask = lbuilder.buildGraphLabeler(opts, dummyBuilder, reprBuilder)
    elif parser == "TRANS" and labeler == "trans":
        lblTask = ltask.DummyLabeler(parsingTask.getTransLabeler())
    elif labeler == None or not labeler or labeler == "None":
        lblTask = ltask.DummyLabeler()
    elif parser == "GRAPH" and labeler == "graph":
        lblTask = ltask.DummyLabeler(parsingTask.getLblDict())
    else:
        logging.error("Unknown labeler: %s for parser %s" % (labeler, parser))
        sys.exit()
        
    return lblTask
