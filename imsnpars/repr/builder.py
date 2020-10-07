'''
Created on 23.08.2018

@author: falensaa
'''

import logging
import sys

from repr import word, sentence, stag, elmo, stack
    
def __buildWordReprBuilder(opts):
    strDropout = str(round(opts.wordDropout, 2)) if opts.wordDropout else "None"
    logging.info("Building WordReprBuilder with wordDim=%i, wordDropout=%s" % (opts.wordDim, strDropout))
    wordBuilder = word.WordReprBuilder(opts.wordDim, opts.wordDropout)
    return wordBuilder

def __buildEWordReprBuilder(opts):
    strDropout = str(round(opts.wordDropout, 2)) if opts.wordDropout else "None"
    logging.info("Building ExtEmbWordReprBuilder with embDim=%i, wordDropout=%s, embUpdate=%s, embLowercased=%s" % (opts.embDim, strDropout, str(opts.embUpdate), str(opts.embLowercased)))
    wordBuilder = word.ExtEmbWordReprBuilder(opts.embDim, opts.wordDropout, opts.embFile, opts.embUpdate, opts.embLowercased)
    return wordBuilder

def __buildPOSReprBuilder(opts):
    logging.info("Building POSReprBuilder with posDim=%i" % opts.posDim)
    posBuilder = word.POSReprBuilder(opts.posDim)
    return posBuilder
        
def __buildCharReprBuilder(opts):
    charDropout = str(round(opts.charDropout, 2)) if opts.charDropout else "None"
    lstmDropout = str(round(opts.charLstmDropout, 2)) if opts.charLstmDropout else "None"
    logging.info("Building CharLstmReprBuilder with charDim=%i, charLstmDim=%i, charDropout=%s, lstmDropout=%s" % (opts.charDim, opts.charLstmDim, charDropout, lstmDropout))
    charBuilder = word.CharLstmReprBuilder(opts.charDim, opts.charLstmDim, charDropout=opts.charDropout, lstmDropout=opts.charLstmDropout)
    return charBuilder
        
def __buildMorphReprBuilder(opts):
    logging.info("Building MorphReprBuilder with morphDim=%i" % opts.morphDim)
    morphBuilder = word.MorphReprBuilder(opts.morphDim)
    return morphBuilder

def __buildSTagReprBuilder(opts):
    logging.info("Building SuperTagReprBuilder with stagDim=%i" % opts.stagDim)
    return stag.SuperTagReprBuilder(opts.stagDim)

def __buildStackLblReprBuilder(opts):
    logging.info("Building StaclLblReprBuilder with lblDim=%i" % (opts.stackLblDim))
    lblBuilder = stack.StackLblReprBuilder(opts.stackLblDim)
    return lblBuilder

def __buildStackHeadReprBuilder(opts):
    strDropout = str(round(opts.wordDropout, 2)) if opts.wordDropout else "None"
    logging.info("Building StackHeadReprBuilder with wordDim=%i, wordDropout=%s" % (opts.stackHeadDim, strDropout))
    headBuilder =  stack.StackHeadReprBuilder(opts.stackHeadDim, opts.wordDropout)
    
    return headBuilder

def __buildStackReprBuilders(opts):
    stackReprBuilders = [  ]
        
    if "lbl" in opts.stackRepr:
        stackReprBuilders.append(__buildStackLblReprBuilder(opts))
            
    if "head" in opts.stackRepr:
        stackReprBuilders.append(__buildStackHeadReprBuilder(opts))
        
    if "stag" in opts.stackRepr:
        stackReprBuilders.append(__buildSTagReprBuilder(opts))
        
    return stackReprBuilders

def __buildContextReprBuilder(opts, tokenBuilder):
    if opts.contextRepr == "bilstm":
        return __buildLSTMReprBuilder(opts, tokenBuilder)
    elif opts.contextRepr == "concat":
        logging.info("Building ConcatReprBuilder")
        return tokenBuilder
    else:
        logging.error("Unknown context representation: " + opts.contextRepr)
        sys.exit()
        
def __buildLSTMReprBuilder(opts, tokenBuilder):
    lstmDropout = str(round(opts.lstmDropout, 2)) if opts.lstmDropout else "None"
    logging.info("Building BiLSTMReprBuilder with lowerLevel=%s, lstmDim=%i, lstmLayers=%i, lstmDropout=%s" % (str(type(tokenBuilder)), opts.lstmDim, opts.lstmLayers, lstmDropout))
    return sentence.BiLSTReprBuilder(tokenBuilder, opts.lstmDim, opts.lstmLayers, lstmDropout=opts.lstmDropout)

def buildReprBuilders(opts):
    reprBuilders = [  ]
    
    if "word" in opts.representations:
        reprBuilders.append(__buildWordReprBuilder(opts))
        
    if "pos" in opts.representations:
        reprBuilders.append(__buildPOSReprBuilder(opts))
        
    if "char" in opts.representations:
        reprBuilders.append(__buildCharReprBuilder(opts))
        
    if "morph" in opts.representations:
        reprBuilders.append(__buildMorphReprBuilder(opts))
    
    if "eword" in opts.representations:
        reprBuilders.append(__buildEWordReprBuilder(opts))
        
    if opts.stackRepr:
        reprBuilders += __buildStackReprBuilders(opts)
        
    return reprBuilders

def buildContextReprBuilder(opts, reprBuilders):
    tokenBuilder = sentence.CollectReprBuilder(reprBuilders)
    
    if "elmo" in opts.representations:
        logging.info("Building ELMoReprBuilder with elmoGamma=%f elmoLearnGamma=%s" % (opts.elmoGamma, str(opts.elmoLearnGamma)))
        tokenBuilder = elmo.ELMoReprBuilder(tokenBuilder, opts.elmoFile, opts.elmoGamma, opts.elmoLearnGamma)
    
    contextReprBuilder = __buildContextReprBuilder(opts, tokenBuilder)
    return contextReprBuilder

