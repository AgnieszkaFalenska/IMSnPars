'''
Created on 23.08.2017

@author: falensaa
'''

from tools import  utils

def addParserCmdArguments(argParser):
    graphArgs = argParser.add_argument_group('Graph-based parser')
    
    # mst algorithm
    graphArgs.add_argument("--mst", help="mst algorithm", choices=[ "CLE" ], required=False, default="CLE")
    graphArgs.add_argument("--augment", help="augment training as described in K&G", choices=[ "True", "False" ], required=False, default="True")
    graphArgs.add_argument("--features", help="graph features (combination of {h,d})", required=False, default="h,d")
    graphArgs.add_argument("--imposeOneRoot", help="only one root in the predicted tree", required=False, default="True")
    
def fillParserOptions(args, opts):
    opts.addOpt("mst", args.mst)
    opts.addOpt("features", args.features.split(","))
    opts.addOpt("augment", utils.parseBoolean(args.augment))
    opts.addOpt("imposeOneRoot", utils.parseBoolean(args.imposeOneRoot))
    
    return opts
