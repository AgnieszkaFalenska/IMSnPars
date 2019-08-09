'''
Created on 23.08.2017

@author: falensaa
'''

#TODO:  add prefix to the options
def addLabelerCmdArguments(argParser):
    lblArgs = argParser.add_argument_group('Labeler')
    
    # type of labeler
    lblArgs.add_argument("--labeler", help="labeler", choices=[ "graph-mtl", "trans-mtl", "trans", "graph", "None" ], required=False, default="None")
    
    # graph-mlt features
    lblArgs.add_argument("--graphLabelerFeats", help="features for the graph mtl labeler", required=False, default="h,d")
    
def fillLabelerOptions(args, options):
    # labeler
    options.labeler = args.labeler if args.labeler != "None" else None
    
    # dimentions
    options.mlpHiddenLblDim = 100    
    options.lblLayer = 1 if args.contextRepr == "bilstm" else None
    
    # features
    if args.labeler == "graph-mtl":
        options.graphLabelerFeats = args.graphLabelerFeats.split(",")
