'''
Created on 15.03.2019

@author: falensaa
'''

import argparse
import random
import logging
import sys

from tools import utils
from nparser import options, builder
import nparser.analysis.tools as atools
 
def buildParserFromArgs():
    argParser = argparse.ArgumentParser(description="""Analyze the IMSnPars parser""")
    argParser.add_argument("--parser", help="which parser [GRAPH|TRANS]", required=True)
    argParser.add_argument("--analyze", help="what to analyze [SURF|FEAT]", required=True)
    argParser.add_argument("--lstmLayer", help="layer for SURF analysis", required=False, type=int, default=2)
    argParser.add_argument("--loglevel", help="which log level to use", choices=[ "DEBUG", "INFO", "WARN", "ERROR", "CRITICAL" ], required=False, default="DEBUG")
    
    # files
    argParser.add_argument("--test", help="test file to analyze", required=True)
    argParser.add_argument("--model", help="load model from file", required=True)
    argParser.add_argument("--output", help="output analysis to file", required=True)
    
    # format
    argParser.add_argument("--format", help="file format", required=False, default="conllu")
    argParser.add_argument("--normalize", help="normalize the words", required=False, default="True")
    
    # dynet and seeds
    argParser.add_argument("--seed", help="random seed (different than dynet-seed)", required=False, default=42, type=int)
    argParser.add_argument("--dynet-gpu-ids", help="turns on dynet", required=False)
    argParser.add_argument("--dynet-devices", help="turns on dynet", required=False)
    argParser.add_argument("--dynet-seed", help="sets random seed generator for dynet", required=False)
    
    
    # parse for the first time to only get the parser name
    args, _ = argParser.parse_known_args()
    
    # set logging level
    lformat = '[%(levelname)s] %(asctime)s %(name)s# %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.getLevelName(args.loglevel), format=lformat)
    
    # parse the second time to get all the args
    options.addParserCmdArguments(args.parser, argParser)
    args = argParser.parse_args()

    # this still leaves dynet-seed being random
    random.seed(args.seed)
    logging.info("Freezing random seed: %i" % args.seed)

    opts = utils.NParserOptions()
    options.fillParserOptions(args, opts)
    
    # all the args fields will get overwritten
    opts.load(args.model + ".args")
    opts.logOptions()
    
    parser = builder.buildParser(opts)
    return args, parser

def buildAnalysisTool(args, parsTask, reprBuilder):
    if args.analyze == "SURF":
        return atools.SurfGradientAnalysis(reprBuilder, args.lstmLayer)
    elif args.analyze == "FEAT":
        if args.parser == "TRANS":
            featAnalyzer = atools.TransFeatAnalyzer(parsTask)
        else:
            featAnalyzer = atools.GraphFeatAnalyzer(parsTask)
        
        return atools.FeatGradientAnalysis(featAnalyzer, reprBuilder)
    
    logging.info("Unknown analysis option: %s" % (args.analyze))
    sys.exit()
    
if __name__ == "__main__":
    args, parser  = buildParserFromArgs()
    
    # format reader
    tokenBuilder = utils.buildFormatReader(args.format)
    normalizer =  builder.buildNormalizer(utils.parseBoolean(args.normalize))
            
    parser.load(args.model)
        
    testData = utils.readSentences(args.test, tokenBuilder, normalizer)
    analysisTool = buildAnalysisTool(args, parser.getParsingTask(), parser.getReprBuilder())
    
    analysisTool.runAnalysis(testData, args.output)