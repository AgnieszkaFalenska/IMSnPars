'''
Created on 23.08.2017

@author: falensaa
'''

def addLabelerCmdArguments(argParser):
    lblArgs = argParser.add_argument_group('Labeler')
    
    # type of labeler
    lblArgs.add_argument("--labeler", help="which labeler to use", choices=[ "graph-mtl", "trans", "graph", "None" ], required=False, default="graph-mtl")
    
    # graph-mtl features
    lblArgs.add_argument("--graphLabelerFeats", help="features for the graph-mtl labeler", required=False, default="h,d")
    
def fillLabelerOptions(args, opts):
    # labeler
    opts.addOpt("labeler", args.labeler if args.labeler != "None" else None)
    
    # dimentions
    opts.addOpt("mlpHiddenLblDim", 100)
    opts.addOpt("lblLayer", 1 if args.contextRepr == "bilstm" else None)
    
    # features
    if args.labeler == "graph-mtl":
        opts.addOpt("graphLabelerFeats", args.graphLabelerFeats.split(","))
