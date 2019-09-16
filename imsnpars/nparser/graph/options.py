'''
Created on 23.08.2017

@author: falensaa
'''

from tools import  utils

def addParserCmdArguments(argParser):
    graphArgs = argParser.add_argument_group('Graph-based parser')
    
    # mst algorithm
    graphArgs.add_argument("--mst", help="mst algorithm", choices=[ "CLE", "Eisner", "EisnerO2sib", "EisnerO2g" ], required=False, default="CLE")
    graphArgs.add_argument("--augment", help="augment training as described in K&G", choices=[ "True", "False" ], required=False, default="True")
    graphArgs.add_argument("--features", help="graph features (combination of {h,d})", required=False, default="h,d")
    
def fillParserOptions(args, opts):
    opts.mst = args.mst
    opts.features = args.features.split(",")
    opts.augment = utils.parseBoolean(args.augment)
    
    return opts
