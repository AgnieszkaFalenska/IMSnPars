'''
Created on 05.04.2019

@author: falensaa
'''

from tools import utils

def addParserCmdArguments(argParser):
    mtlArgs = argParser.add_argument_group('MTL parser')
    mtlArgs.add_argument("--firstTask", help="firstTask", choices=[ "TRANS", "GRAPH" ], required=False)
    mtlArgs.add_argument("--secondTask", help="secondTask", choices=[ "TRANS", "GRAPH" ], required=False)
    mtlArgs.add_argument("--imposeOneRoot", help="only one root in the predicted tree", required=False, default="True")
    
    # labelers
    mtlArgs.add_argument("--t1Labeler", help="labeler for the first task", choices=[ "graph-mtl", "trans", "graph", "None" ], required=False, default="None")
    mtlArgs.add_argument("--t2Labeler", help="labeler for the second task", choices=[ "graph-mtl", "trans", "graph", "None" ], required=False, default="None")
    
    # first transition-based task
    secondTransArgs = argParser.add_argument_group('First transition-based parser')
    secondTransArgs.add_argument("--t1Reversed", help="reverse sentences", choices=[ "True", "False" ], required=False, default="False")
    secondTransArgs.add_argument("--t1System", help="transition system", choices=[ "ArcHybrid", "ArcStandard", "ASSwap", "ArcHybridWithSwap" ], required=False, default="ASSwap")
    secondTransArgs.add_argument("--t1Oracle", help="oracle", choices=[ "static", "dynamic", "lazy", "eager" ], required=False, default="lazy")
    secondTransArgs.add_argument("--t1Features", help="transition-based features", required=False, default="s0,s1,b0")
    
    
    # second transition-based task
    secondTransArgs = argParser.add_argument_group('Second transition-based parser')
    secondTransArgs.add_argument("--t2Reversed", help="reverse sentences", choices=[ "True", "False" ], required=False, default="False")
    secondTransArgs.add_argument("--t2System", help="transition system", choices=[ "ArcHybrid", "ArcStandard", "ASSwap", "ArcHybridWithSwap" ], required=False, default="ASSwap")
    secondTransArgs.add_argument("--t2Oracle", help="oracle", choices=[ "static", "dynamic", "lazy", "eager" ], required=False, default="lazy")
    secondTransArgs.add_argument("--t2Features", help="transition-based features", required=False, default="s0,s1,b0")
    
    
def fillParserOptions(args, opts):
    opts.addOpt("firstTask", args.firstTask)
    opts.addOpt("secondTask", args.secondTask)
    opts.addOpt("imposeOneRoot", utils.parseBoolean(args.imposeOneRoot))
    
    opts.addOpt("t1Labeler", args.t1Labeler)
    opts.addOpt("t2Labeler", args.t2Labeler)
    
    if opts.firstTask == "GRAPH": 
        # for now default values since I don't options here
        opts.addOpt("t1Mst", "CLE")
        
        # the features might be still the default values
        if args.t1Features != "s0,s1,b0":
            opts.addOpt("t1Features", args.t1Features.split(","))
        else:
            opts.addOpt("t1Features", "h,d".split(","))
            
        opts.addOpt("t1Augment", True)
    else:
        opts.addOpt("t1Reversed", utils.parseBoolean(args.t1Reversed))
        opts.addOpt("t1System", args.t1System)
        opts.addOpt("t1Oracle", args.t1Oracle)
        opts.addOpt("t1Features", args.t1Features.split(","))
        
    if opts.secondTask == "GRAPH":
        # for now default values since I don't options here
        opts.addOpt("t2Mst", "CLE")
        
        # the features might be still the default values
        if args.t2Features != "s0,s1,b0":
            opts.addOpt("t2Features", args.t2Features.split(","))
        else:
            opts.addOpt("t2Features", "h,d".split(","))
            
        opts.addOpt("t2Augment", True)
    else:
        opts.addOpt("t2Reversed", utils.parseBoolean(args.t2Reversed))
        opts.addOpt("t2System", args.t2System)
        opts.addOpt("t2Oracle", args.t2Oracle)
        opts.addOpt("t2Features", args.t2Features.split(","))
    
    return opts
