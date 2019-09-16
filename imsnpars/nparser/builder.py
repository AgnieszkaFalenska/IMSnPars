'''
Created on 23.08.2017

@author: falensaa
'''

import logging
import dynet
import sys

from tools import utils
from repr import word, sentence
    
from nparser.task import NDependencyParser
from nparser.features import ZeroDummyVectorBuilder, RandomDummyVectorBuilder

import nparser.graph.builder as gbuilder
import nparser.trans.builder as tbuilder
import nparser.labels.builder as lbuilder
import nparser.labels.task as ltask

def buildNormalizer(normalize):
    if normalize:
        return utils.Normalizer(normalizeNumbers = True, lowercase = True)
    else:
        return None
        
def buildParser(opts):
    reprBuilder, dummyBuilder = __buildReprBuilder(opts)
    trainer = __buildTrainer(opts)
    
    if opts.parser == "TRANS":
        parsingTask = tbuilder.buildTransParser(opts, dummyBuilder, reprBuilder)
        lblTask = __buildLabeler(opts.labeler, opts.parser, opts, dummyBuilder, reprBuilder, parsingTask)
        
    elif opts.parser == "GRAPH":
        parsingTask = gbuilder.buildGraphParser(opts, dummyBuilder, reprBuilder)
        lblTask = __buildLabeler(opts.labeler, opts.parser, opts, dummyBuilder, reprBuilder, parsingTask)
        
    else:
        logging.error("Unknown parser: %s" % opts.parser)
        sys.exit()
    
    return NDependencyParser(reprBuilder, parsingTask, lblTask, trainer)

def __buildWordReprBuilder(opts):
    strDropout = str(round(opts.wordDropout, 2)) if opts.wordDropout else "None"
    logging.info("Building WordReprBuilder with wDim=%i, wordDropout=%s" % (opts.wDim, strDropout))
    wordBuilder = word.WordReprBuilder(opts.wDim, opts.wordDropout)
    return wordBuilder

def __buildPOSReprBuilder(opts):
    logging.info("Building POSReprBuilder with posDim=%i" % opts.posDim)
    posBuilder = word.POSReprBuilder(opts.posDim)
    return posBuilder
        
def __buildCharReprBuilder(opts):
    charDropout = str(round(opts.charDropout, 2)) if opts.charDropout else "None"
    lstmDropout = str(round(opts.lstmDropout, 2)) if opts.lstmDropout else "None"
    logging.info("Building CharLstmReprBuilder with charDim=%i, charLstmDim=%i, charDropout=%s, lstmDropout=%s" % (opts.charDim, opts.charLstmDim, charDropout, lstmDropout))
    charBuilder = word.CharLstmReprBuilder(opts.charDim, opts.charLstmDim, charDropout=opts.charDropout, lstmDropout=opts.lstmDropout)
    return charBuilder
        
def __buildMorphReprBuilder(opts):
    logging.info("Building MorphReprBuilder with morphDim=%i" % opts.morphDim)
    morphBuilder = word.MorphReprBuilder(opts.morphDim)
    return morphBuilder
        
def __buildContextReprBuilder(opts, tokenBuilder):
    if opts.contextRepr == "bilstm":
        lstmDropout = str(round(opts.lstmDropout, 2)) if opts.lstmDropout else "None"
        logging.info("Building BiLSTMReprBuilder with lstmDim=%i, lstmLayers=%i, lstmDropout=%s" % (opts.lstmDim, opts.lstmLayers, lstmDropout))
        vecBuilder = sentence.BiLSTReprBuilder(tokenBuilder, opts.lstmDim, opts.lstmLayers, lstmDropout=opts.lstmDropout)
    elif opts.contextRepr == "concat":
        logging.info("Building ConcatReprBuilder")
        vecBuilder = tokenBuilder
    else:
        logging.error("Unknown context representation: " + opts.contextRepr)
        sys.exit()
        
    return vecBuilder
        
def __buildDummyReprBuilder(opts, reprDim, outDim):
    if opts.dummy == "zero":
        return ZeroDummyVectorBuilder(outDim)
    elif opts.dummy == "random":
        return RandomDummyVectorBuilder(reprDim, outDim, opts.nonLinFun, separate = False)
    elif opts.dummy == "random_sep":
        return RandomDummyVectorBuilder(reprDim, outDim, opts.nonLinFun, separate = True)
    else:
        logging.error("Unknown dummy builder: %s" % opts.dummy)
        sys.exit()
        
def __buildReprBuilder(opts):
    reprBuilders = [  ]
    
    if "word" in opts.representations:
        reprBuilders.append(__buildWordReprBuilder(opts))
        
    if "pos" in opts.representations:
        reprBuilders.append(__buildPOSReprBuilder(opts))
        
    if "char" in opts.representations:
        reprBuilders.append(__buildCharReprBuilder(opts))
        
    if "morph" in opts.representations:
        reprBuilders.append(__buildMorphReprBuilder(opts))
    
    tokenBuilder = sentence.CollectReprBuilder(reprBuilders)
    contextReprBuilder = __buildContextReprBuilder(opts, tokenBuilder)
        
    dummyBuilder = __buildDummyReprBuilder(opts, sum([ rd.getDim() for rd in reprBuilders ]), contextReprBuilder.getDim())
    logging.info("Using dummy builder: %s" % type(dummyBuilder))
    return contextReprBuilder, dummyBuilder

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
        logging.info("Building DummyLabeler")
        lblTask = ltask.DummyLabeler()
    elif parser == "GRAPH" and labeler == "graph":
        lblTask = ltask.DummyLabeler(parsingTask.getLblDict())
    else:
        logging.error("Unknown labeler: %s for parser %s" % (labeler, parser))
        sys.exit()
        
    return lblTask
