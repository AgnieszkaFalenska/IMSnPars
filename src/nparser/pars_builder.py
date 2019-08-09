'''
Created on 23.08.2017

@author: falensaa
'''

import logging
import dynet
import sys

from tools import utils
from repr.word_repr import WordReprBuilder, POSReprBuilder, CharLstmReprBuilder, MorphReprBuilder
from repr.sent_repr import CollectReprBuilder, BiLSTReprBuilder
    
from nparser.pars_main import NDependencyParser
from nparser.graph import gp_builder
from nparser.labels import lbl_task, lbl_builder
from nparser.trans import tp_builder
from nparser.pars_features import ZeroDummyVectorBuilder, RandomDummyVectorBuilder

def buildNormalizer(normalize):
    if normalize:
        return utils.Normalizer(normalizeNumbers = True, lowercase = True)
    else:
        return None
        
def buildParser(options):
    reprBuilder, dummyBuilder = __buildReprBuilder(options)
    trainer = __buildTrainer(options)
    
    if options.parser == "TRANS":
        parsingTask = tp_builder.buildTransParser(options, dummyBuilder, reprBuilder)
        lblTask = __buildLabeler(options.labeler, options.parser, options, dummyBuilder, reprBuilder, parsingTask)
        
    elif options.parser == "GRAPH":
        parsingTask = gp_builder.buildGraphParser(options, dummyBuilder, reprBuilder)
        lblTask = __buildLabeler(options.labeler, options.parser, options, dummyBuilder, reprBuilder, parsingTask)
        
    else:
        logging.error("Unknown parser: %s" % options.parser)
        sys.exit()
    
    return NDependencyParser(reprBuilder, parsingTask, lblTask, trainer)

def __buildWordReprBuilder(options):
    strDropout = str(round(options.wordDropout, 2)) if options.wordDropout else "None"
    logging.info("Building WordReprBuilder with wDim=%i, wordDropout=%s" % (options.wDim, strDropout))
    wordBuilder = WordReprBuilder(options.wDim, options.wordDropout)
    return wordBuilder

def __buildPOSReprBuilder(options):
    logging.info("Building POSReprBuilder with posDim=%i" % options.posDim)
    posBuilder = POSReprBuilder(options.posDim)
    return posBuilder
        
def __buildCharReprBuilder(options):
    charDropout = str(round(options.charDropout, 2)) if options.charDropout else "None"
    lstmDropout = str(round(options.lstmDropout, 2)) if options.lstmDropout else "None"
    logging.info("Building CharLstmReprBuilder with charDim=%i, charLstmDim=%i, charDropout=%s, lstmDropout=%s" % (options.charDim, options.charLstmDim, charDropout, lstmDropout))
    charBuilder = CharLstmReprBuilder(options.charDim, options.charLstmDim, charDropout=options.charDropout, lstmDropout=options.lstmDropout)
    return charBuilder
        
def __buildMorphReprBuilder(options):
    logging.info("Building MorphReprBuilder with morphDim=%i" % options.morphDim)
    morphBuilder = MorphReprBuilder(options.morphDim)
    return morphBuilder
        
def __buildContextReprBuilder(options, tokenBuilder):
    if options.contextRepr == "bilstm":
        lstmDropout = str(round(options.lstmDropout, 2)) if options.lstmDropout else "None"
        logging.info("Building BiLSTMReprBuilder with lstmDim=%i, lstmLayers=%i, lstmDropout=%s" % (options.lstmDim, options.lstmLayers, lstmDropout))
        vecBuilder = BiLSTReprBuilder(tokenBuilder, options.lstmDim, options.lstmLayers, lstmDropout=options.lstmDropout)
    elif options.contextRepr == "concat":
        logging.info("Building ConcatReprBuilder")
        vecBuilder = tokenBuilder
    else:
        logging.error("Unknown context representation: " + options.contextRepr)
        sys.exit(-1)
        
    return vecBuilder
        
def __buildDummyReprBuilder(options, reprDim, outDim):
    if options.dummy == "zero":
        return ZeroDummyVectorBuilder(outDim)
    elif options.dummy == "random":
        return RandomDummyVectorBuilder(reprDim, outDim, options.nonLinFun, separate = False)
    elif options.dummy == "random_sep":
        return RandomDummyVectorBuilder(reprDim, outDim, options.nonLinFun, separate = True)
    else:
        logging.error("Unknown dummy builder: %s" % options.dummy)
        sys.exit()
        
def __buildReprBuilder(options):
    reprBuilders = [  ]
    
    if "word" in options.representations:
        reprBuilders.append(__buildWordReprBuilder(options))
        
    if "pos" in options.representations:
        reprBuilders.append(__buildPOSReprBuilder(options))
        
    if "char" in options.representations:
        reprBuilders.append(__buildCharReprBuilder(options))
        
    if "morph" in options.representations:
        reprBuilders.append(__buildMorphReprBuilder(options))
    
    tokenBuilder = CollectReprBuilder(reprBuilders)
    contextReprBuilder = __buildContextReprBuilder(options, tokenBuilder)
        
    dummyBuilder = __buildDummyReprBuilder(options, sum([ rd.getDim() for rd in reprBuilders ]), contextReprBuilder.getDim())
    logging.info("Using dummy builder: %s" % type(dummyBuilder))
    return contextReprBuilder, dummyBuilder

def __buildTrainer(options):
    if options.trainer == "adam":
        trainerClass = dynet.AdamTrainer
    elif options.trainer == "simple-sgd":
        trainerClass = dynet.SimpleSGDTrainer
    elif options.trainer == "momentum-sgd":
        trainerClass = dynet.MomentumSGDTrainer
        
    if options.learningRate is not None:
        if options.trainer == "adam":
            trainer = lambda model : trainerClass(model, alpha=options.learningRate)
        else:
            trainer = lambda model : trainerClass(model, learning_rate=options.learningRate)
    else:
        trainer = trainer = lambda model : trainerClass(model)
          
    logging.info("Building trainer: %s with learning rate: %s" % ( trainerClass, "'default'" if not options.learningRate else str(options.learningRate)))  
    return trainer


def __buildLabeler(labeler, parser, moptions, dummyBuilder, reprBuilder, parsingTask):
    if labeler == "graph-mtl":
        lblTask = lbl_builder.buildGraphLabeler(moptions, dummyBuilder, reprBuilder)
    elif labeler == "trans-mtl":
        lblTask = lbl_builder.buildTransLabeler(moptions, dummyBuilder, reprBuilder)
    elif parser == "TRANS" and labeler == "trans":
        lblTask = lbl_task.DummyLabeler(parsingTask.getTransLabeler())
    elif labeler == None or not labeler or labeler == "None":
        logging.info("Building DummyLabeler")
        lblTask = lbl_task.DummyLabeler()
    elif parser == "GRAPH" and labeler == "graph":
        lblTask = lbl_task.DummyLabeler(parsingTask.getLblDict())
    else:
        logging.error("Unknown labeler: %s for parser %s" % (labeler, parser))
        sys.exit()
        
    return lblTask
