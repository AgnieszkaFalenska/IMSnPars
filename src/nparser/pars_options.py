'''
Created on 23.08.2017

@author: falensaa
'''

import logging
import dynet
import sys

from tools import utils
from nparser.trans import tp_options
from nparser.graph import gp_options
from nparser.labels import lbl_options

def addCommonParserCmdArguments(argParser):
    # word representations
    reprArgs = argParser.add_argument_group('Token representations')
    reprArgs.add_argument("--contextRepr", help="context representation", choices=[ "bilstm", "concat" ], required=False, default="bilstm")
    reprArgs.add_argument("--dummy", help="which dummy repr to use", choices=[ "zero", "random", "random_sep" ], required=False, default="random")
    reprArgs.add_argument("--representations", help="which representations to use, select a subset of [word|pos|char|morph], use ',' separator", required=False, default="word,pos,char")
    
    # char representations
    reprArgs.add_argument("--charDropout", help="char Dropout", required=False, default="0.25")
    reprArgs.add_argument("--charDim", help="charDim", required=False, default=24, type=int)
    reprArgs.add_argument("--charLstmDim", help="charLstmDim", required=False, default=100, type=int)
    
    # word representations
    reprArgs.add_argument("--wordDropout", help="word Dropout", required=False, default="0.25")
    
    # bilstm features
    bilstmArgs = argParser.add_argument_group('BiLSTM')
    bilstmArgs.add_argument("--parseLayer", help="layer of bilstms to use for parsing", required=False, type=int, default=1)
    bilstmArgs.add_argument("--lstmLayers", help="nr of layers", required=False, type=int, default=2)
    bilstmArgs.add_argument("--lstmDim", help="lstmDim", required=False, default=125, type=int)
    bilstmArgs.add_argument("--lstmDropout", help="lstm Dropout", required=False, default="0.33")
    
    # training
    trainArgs = argParser.add_argument_group('Parser training')
    trainArgs.add_argument("--trainer", help="which trainer to use", required=False, default="adam")
    trainArgs.add_argument("--learningRate", help="learning rate", required=False, default="None")
    
def addParserCmdArguments(parser, argParser):
    addCommonParserCmdArguments(argParser)
    
    if parser == "TRANS":
        tp_options.addParserCmdArguments(argParser)
    elif parser == "GRAPH":
        gp_options.addParserCmdArguments(argParser)
    else:
        logging.error("Unknown parser: %s" % parser)
        sys.exit()
      
    lbl_options.addLabelerCmdArguments(argParser)

def fillCommonParserOptions(args, options):
    
    ## name of the parser
    options.parser = args.parser
    options.normalize = args.normalize
    
    # lstms and representations
    options.mlpHiddenDim = 100
    options.lstmDim = args.lstmDim
    options.lstmDropout = float(args.lstmDropout) if args.lstmDropout != "None" else None
    
    ### layers
    options.lstmLayers = args.lstmLayers
    options.parseLayer = args.parseLayer if args.contextRepr == "bilstm" else None
    
    ### words
    options.wDim = 100
    options.wordDropout = float(args.wordDropout) if args.wordDropout != "None" else None
    
    ### dimentions
    options.posDim = 20
    options.morphDim = 50
    options.lblDim = 20
    
    ### char
    options.charDim = args.charDim
    options.charLstmDim = args.charLstmDim
    options.charDropout = float(args.charDropout) if args.charDropout != "None" else None
    
    # network training
    options.trainMargin = 1.0
    options.learningRate = float(args.learningRate) if args.learningRate != "None" else None
    options.trainer = args.trainer
    options.nonLinFun = dynet.tanh
    
    # representations
    options.dummy = args.dummy
    options.representations = args.representations.split(",")
    options.contextRepr = args.contextRepr
    return options

def fillParserOptions(args, options):
    fillCommonParserOptions(args, options)
    
    if args.parser == "TRANS":
        options = tp_options.fillParserOptions(args, options)
    elif args.parser == "GRAPH":
        options = gp_options.fillParserOptions(args, options)
    else:
        logging.error("Unknown parser: %s" % args.parser)
        sys.exit()
    
    lbl_options.fillLabelerOptions(args, options)
    return options

