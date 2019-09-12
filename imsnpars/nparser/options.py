'''
Created on 23.08.2017

@author: falensaa
'''

import logging
import dynet
import sys

import nparser.trans.options as toptions
import nparser.graph.options as goptions
import nparser.labels.options as loptions

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
        toptions.addParserCmdArguments(argParser)
    elif parser == "GRAPH":
        goptions.addParserCmdArguments(argParser)
    else:
        logging.error("Unknown parser: %s" % parser)
        sys.exit()
      
    loptions.addLabelerCmdArguments(argParser)

def fillCommonParserOptions(args, opts):
    
    ## name of the parser
    opts.parser = args.parser
    opts.normalize = args.normalize
    
    # lstms and representations
    opts.mlpHiddenDim = 100
    opts.lstmDim = args.lstmDim
    opts.lstmDropout = float(args.lstmDropout) if args.lstmDropout != "None" else None
    
    ### layers
    opts.lstmLayers = args.lstmLayers
    opts.parseLayer = args.parseLayer if args.contextRepr == "bilstm" else None
    
    ### words
    opts.wDim = 100
    opts.wordDropout = float(args.wordDropout) if args.wordDropout != "None" else None
    
    ### dimentions
    opts.posDim = 20
    opts.morphDim = 50
    opts.lblDim = 20
    
    ### char
    opts.charDim = args.charDim
    opts.charLstmDim = args.charLstmDim
    opts.charDropout = float(args.charDropout) if args.charDropout != "None" else None
    
    # network training
    opts.trainMargin = 1.0
    opts.learningRate = float(args.learningRate) if args.learningRate != "None" else None
    opts.trainer = args.trainer
    opts.nonLinFun = dynet.tanh
    
    # representations
    opts.dummy = args.dummy
    opts.representations = args.representations.split(",")
    opts.contextRepr = args.contextRepr
    return opts

def fillParserOptions(args, opts):
    fillCommonParserOptions(args, opts)
    
    if args.parser == "TRANS":
        opts = toptions.fillParserOptions(args, opts)
    elif args.parser == "GRAPH":
        opts = goptions.fillParserOptions(args, opts)
    else:
        logging.error("Unknown parser: %s" % args.parser)
        sys.exit()
    
    loptions.fillLabelerOptions(args, opts)
    return opts

