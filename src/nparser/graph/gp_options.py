'''
Created on 23.08.2017

@author: falensaa
'''

import argparse
from tools import  utils

def addParserCmdArguments(argParser):
    graphArgs = argParser.add_argument_group('Graph-based parser')
    
    # mst algorithm
    graphArgs.add_argument("--mst", help="mst algorithm", choices=[ "CLE" ], required=False, default="CLE")
    graphArgs.add_argument("--augment", help="augment training as described in K&G", choices=[ "True", "False" ], required=False, default="True")
    graphArgs.add_argument("--features", help="graph features (combination of {h,d})", required=False, default="h,d")
    
def fillParserOptions(args, options):
    options.mst = args.mst
    options.features = args.features.split(",")
    options.augment = utils.parseBoolean(args.augment)
    
    return options