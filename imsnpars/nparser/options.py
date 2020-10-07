'''
Created on 23.08.2017

@author: falensaa
'''

import logging
import dynet
import sys

import repr.options as roptions
import nparser.trans.options as toptions
import nparser.graph.options as goptions
import nparser.labels.options as loptions

def addCommonParserCmdArguments(argParser):
    # features
    generalArgs = argParser.add_argument_group('NParser -- general')
    generalArgs.add_argument("--dummy", help="which dummy feature builder to use", choices=[ "zero", "random", "random_sep" ], required=False, default="random")
    generalArgs.add_argument("--parseLayer", help="layer of bilstms to use for parsing", required=False, type=int, default=1)
    
    # training
    trainArgs = argParser.add_argument_group('NParser -- training')
    trainArgs.add_argument("--trainer", help="which trainer to use", required=False, default="adam")
    trainArgs.add_argument("--learningRate", help="learning rate", required=False, default="default")
    trainArgs.add_argument("--mlpHiddenDim", help="dimensionality of the hidden layer in MLP", required=False, type=int, default=100)
    trainArgs.add_argument("--trainMargin", help="margin for the loss function", required=False, type=float, default=1.0)
    
    
    
def addParserCmdArguments(parser, argParser):
    addCommonParserCmdArguments(argParser)
    roptions.addReprCmdArguments(argParser)
    
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
    opts.addOpt("parser", args.parser)
    
    # general parser options
    opts.addOpt("dummy", args.dummy) # feature representations
    opts.addOpt("parseLayer", args.parseLayer if args.contextRepr == "bilstm" else None)
    
    # network training
    opts.addOpt("trainer", args.trainer)
    opts.addOpt("learningRate", float(args.learningRate) if args.learningRate != "default" else None)
    opts.addOpt("mlpHiddenDim", args.mlpHiddenDim)
    opts.addOpt("trainMargin", args.trainMargin)
    opts.addOpt("nonLinFun", dynet.tanh)
    
    return opts

def fillParserOptions(args, opts):
    fillCommonParserOptions(args, opts)
    roptions.fillReprOptions(args, opts)
    
    if args.parser == "TRANS":
        opts = toptions.fillParserOptions(args, opts)
    elif args.parser == "GRAPH":
        opts = goptions.fillParserOptions(args, opts)
    else:
        logging.error("Unknown parser: %s" % args.parser)
        sys.exit()
    
    loptions.fillLabelerOptions(args, opts)
    return opts

