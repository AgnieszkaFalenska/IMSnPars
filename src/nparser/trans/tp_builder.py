'''
Created on 23.08.2017

@author: falensaa
'''

import logging
import sys

from nparser.pars_network import ParserNetwork
from nparser.trans.tp_task import NNTransParsingTask
from nparser.trans import archybrid, oracle, ahswap, arcstandard, asswap
from nparser.trans.tp_labeler import TransSystemLabeler
from nparser.trans.tp_features import FeatId, StateFeatExtractor,TransFeatureExtractor
from nparser.pars_features import FeatReprBuilder
         
def _buildOracleAndTransSystem(options):
    tsystem = _buildTransSystem(options)
    
    if options.labeler == "trans":
        transLabeler = TransSystemLabeler(tsystem)
    else:
        transLabeler = None
    
    anoracle = _buildOracle(options, tsystem, transLabeler)
         
    logging.info("Trans system used: %s" % type(tsystem))
    logging.info("Oracle used: %s" % type(anoracle))
    logging.info("Trans-labeler used: %s" % type(transLabeler))
        
    if transLabeler != None:
        tsystem = transLabeler
        
    return tsystem, anoracle

def _buildOracle(options, tsystem, transLabeler):
    if options.system != "ArcHybrid" and options.oracle == "dynamic":
        logging.error("No dynamic oracle implemented for %s" % options.system)
        sys.exit(-1)
        
    elif options.system != "ASSwap" and options.oracle in [ "lazy", "eager" ]:
        logging.error("Oracle %s implemented only for system 'ASSwap'" % options.oracle)
        sys.exit(-1)
     
    if options.oracle == "dynamic":
        if options.aggresive:
            policy = oracle.AggresiveExplorePolicy(options.pagg)
        else:
            policy = oracle.OriginalExplorePolicy(firstCorrect=1, predProp=0.9)
             
        logging.info("Explore policy: %s" % type(policy))
        anoracle = archybrid.ArcHybridDynamicOracle(tsystem, labeler=transLabeler, policy=policy)
        
    elif options.oracle == "lazy":
        anoracle = asswap.ArcStandardWithSwapLazyOracle(tsystem, labeler=transLabeler)
    elif options.oracle in [ "eager", "static" ] and options.system == "ASSwap" :
        anoracle = asswap.ArcStandardWithSwapEagerOracle(tsystem, labeler=transLabeler)
    elif options.system == "ArcStandard":
        anoracle = arcstandard.ArcStandardStaticOracle(tsystem, labeler=transLabeler)
    elif options.system == "ArcHybrid":
        anoracle = archybrid.ArcHybridStaticOracle(tsystem, labeler=transLabeler)
    elif options.system == "ArcHybridWithSwap" and options.oracle == "static":
        anoracle = ahswap.ArcHybridWithSwapStaticOracle(tsystem, labeler=transLabeler)
    else:
        logging.error("This setting is not known: oracle=%s, system=%s" % (options.oracle, options.system))
        sys.exit(-1)
        
    return anoracle
        
def _buildTransSystem(options):
    if options.system == "ArcHybrid":
        return archybrid.ArcHybrid()
    elif options.system == "ArcStandard":
        return arcstandard.ArcStandard()
    elif options.system == "ASSwap":
        return asswap.ArcStandardWithSwap()
    elif options.system == "ArcHybridWithSwap":
        return ahswap.ArcHybridWithSwap()
    else:
        logging.error("Unknown transition system: %s" % options.system)
        sys.exit(-1)
   
allFeatIds = { "s0": FeatId.S0,
                "s1" : FeatId.S1,
                "s2" : FeatId.S2,
                "b0" : FeatId.B0,
                "b1" : FeatId.B1,
                "b2" : FeatId.B2,
                "s0lmc" : FeatId.S0LMC,
                "s0rmc" : FeatId.S0RMC,
                "s1lmc" : FeatId.S1LMC,
                "s1rmc" : FeatId.S1RMC,
                "s2lmc" : FeatId.S2LMC,
                "s2rmc" : FeatId.S2RMC,
                "b0lmc" : FeatId.B0LMC,
                "b0rmc" : FeatId.B0RMC,
                "s0lmc_2" : FeatId.S0LMC_2,
                "s0rmc_2" : FeatId.S0RMC_2,
                "s0lmc_lmc" : FeatId.S0LMC_LMC,
                "s0rmc_rmc" : FeatId.S0RMC_RMC,
                "s1lmc_lmc" : FeatId.S1LMC_LMC,
                "s1rmc_rmc" : FeatId.S1RMC_RMC }

def _buildFeatureExtractor(features):
    stateExtractors = [ ]
    
    for feat in features:
        featId = allFeatIds.get(feat)
        if featId == None:
            logging.error("Unknown feat id: %s" % feat)
            sys.exit()
            
        stateExtractors.append(StateFeatExtractor(featId))
            
    extractor = TransFeatureExtractor(stateExtractors)
    return extractor

def buildTransParser(options, dummyBuilder, reprBuilder):
    tsystem, anoracle = _buildOracleAndTransSystem(options)
    transExtractor = _buildFeatureExtractor(options.features)
    
    featIds = transExtractor.getFeatIds()
    transNetwork = ParserNetwork(options.mlpHiddenDim, options.nonLinFun, featIds)
    featBuilder = FeatReprBuilder(transExtractor, { }, dummyBuilder, transNetwork, options.parseLayer)
    
    parsingTask = NNTransParsingTask(tsystem, anoracle, transNetwork, featBuilder)
    return parsingTask
