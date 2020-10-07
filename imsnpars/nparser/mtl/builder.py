'''
Created on 23.08.2017

@author: falensaa
'''

import logging

from nparser.trans import builder as tbuilder 
from nparser.graph import builder as gbuilder

def buildMTLParser(opts, dummyBuilder, reprBuilder):
    opts.labeler = opts.t1Labeler
    opts.features = opts.t1Features
    
    logging.info("Building the first task [%s]" % opts.firstTask)
    if opts.firstTask == "TRANS":
        opts.reversed = opts.t1Reversed
        opts.system = opts.t1System
        opts.oracle = opts.t1Oracle    
        firstTask = tbuilder.buildTransParser(opts, dummyBuilder, reprBuilder)
    else:
        opts.mst = opts.t1Mst
        opts.augment = opts.t1Features
        firstTask = gbuilder.buildGraphParser(opts, dummyBuilder, reprBuilder)
    
    opts.labeler = opts.t2Labeler
    opts.features = opts.t2Features
    
    logging.info("Building the second task [%s]" % opts.secondTask)
    if opts.secondTask == "TRANS":
        opts.reversed = opts.t2Reversed
        opts.system = opts.t2System
        opts.oracle = opts.t2Oracle
        secondTask = tbuilder.buildTransParser(opts, dummyBuilder, reprBuilder)
    else:
        opts.mst = opts.t2Mst
        opts.augment = opts.t2Augment
        secondTask = gbuilder.buildGraphParser(opts, dummyBuilder, reprBuilder)
        
    logging.info("Two tasks: %s, %s" % (type(firstTask), type(secondTask)))
    return firstTask, secondTask