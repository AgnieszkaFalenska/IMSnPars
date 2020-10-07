'''
Created on 23.08.2017

@author: falensaa
'''

import logging
import sys

import nparser.network
import nparser.features
import nparser.trans.features as tfeatures
from nparser.trans import task, labeler
from nparser.trans.tsystem import asswap, arcstandard, archybrid, ahswap, oracle

def _buildOracleAndTransSystem(opts):
    tsystem = _buildTransSystem(opts)
    
    if opts.labeler == "trans":
        transLabeler = labeler.TransSystemLabeler(tsystem)
    else:
        transLabeler = None
    
    anoracle = _buildOracle(opts, tsystem, transLabeler)
         
    logging.info("Trans system used: %s" % type(tsystem))
    logging.info("Oracle used: %s" % type(anoracle))
        
    if transLabeler != None:
        logging.info("Trans-labeler used: %s" % type(transLabeler))
        tsystem = transLabeler
        
    return tsystem, anoracle

def _buildOracle(opts, tsystem, transLabeler):
    if opts.system != "ArcHybrid" and opts.oracle == "dynamic":
        logging.error("No dynamic oracle implemented for %s" % opts.system)
        sys.exit()
        
    elif opts.system != "ASSwap" and opts.oracle in [ "lazy", "eager" ]:
        logging.error("Oracle %s implemented only for system 'ASSwap'" % opts.oracle)
        sys.exit()
     
    if opts.oracle == "dynamic":
        if opts.aggresive:
            policy = oracle.AggresiveExplorePolicy(opts.pagg)
        else:
            policy = oracle.OriginalExplorePolicy(firstCorrect=1, predProp=0.9)
             
        logging.info("Explore policy: %s" % type(policy))
        anoracle = archybrid.ArcHybridDynamicOracle(tsystem, labeler=transLabeler, policy=policy)
        
    elif opts.oracle == "lazy":
        anoracle = asswap.ArcStandardWithSwapLazyOracle(tsystem, labeler=transLabeler)
    elif opts.oracle in [ "eager", "static" ] and opts.system == "ASSwap" :
        anoracle = asswap.ArcStandardWithSwapEagerOracle(tsystem, labeler=transLabeler)
    elif opts.system == "ArcStandard":
        anoracle = arcstandard.ArcStandardStaticOracle(tsystem, labeler=transLabeler)
    elif opts.system == "ArcHybrid":
        anoracle = archybrid.ArcHybridStaticOracle(tsystem, labeler=transLabeler)
    elif opts.system == "ArcHybridWithSwap" and opts.oracle == "static":
        anoracle = ahswap.ArcHybridWithSwapStaticOracle(tsystem, labeler=transLabeler)
    else:
        logging.error("This setting is not known: oracle=%s, system=%s" % (opts.oracle, opts.system))
        sys.exit()
        
    return anoracle
        
def _buildTransSystem(opts):
    if opts.system == "ArcHybrid":
        return archybrid.ArcHybrid()
    elif opts.system == "ArcStandard":
        return arcstandard.ArcStandard()
    elif opts.system == "ASSwap":
        return asswap.ArcStandardWithSwap()
    elif opts.system == "ArcHybridWithSwap":
        return ahswap.ArcHybridWithSwap()
    else:
        logging.error("Unknown transition system: %s" % opts.system)
        sys.exit()
   
allFeatIds = { "s0": tfeatures.FeatId.S0,
                "s1" : tfeatures.FeatId.S1,
                "s2" : tfeatures.FeatId.S2,
                "b0" : tfeatures.FeatId.B0,
                "b1" : tfeatures.FeatId.B1,
                "b2" : tfeatures.FeatId.B2,
                "s0lmc" : tfeatures.FeatId.S0LMC,
                "s0rmc" : tfeatures.FeatId.S0RMC,
                "s1lmc" : tfeatures.FeatId.S1LMC,
                "s1rmc" : tfeatures.FeatId.S1RMC,
                "s2lmc" : tfeatures.FeatId.S2LMC,
                "s2rmc" : tfeatures.FeatId.S2RMC,
                "b0lmc" : tfeatures.FeatId.B0LMC,
                "b0rmc" : tfeatures.FeatId.B0RMC,
                "s0lmc_2" : tfeatures.FeatId.S0LMC_2,
                "s0rmc_2" : tfeatures.FeatId.S0RMC_2,
                "s0lmc_lmc" : tfeatures.FeatId.S0LMC_LMC,
                "s0rmc_rmc" : tfeatures.FeatId.S0RMC_RMC,
                "s1lmc_lmc" : tfeatures.FeatId.S1LMC_LMC,
                "s1rmc_rmc" : tfeatures.FeatId.S1RMC_RMC }

def _buildFeatureExtractor(featuresD):
    stateExtractors = [ ]
    
    for feat in featuresD:
        featId = allFeatIds.get(feat)
        if featId == None:
            logging.error("Unknown feat id: %s" % feat)
            sys.exit()
            
        stateExtractors.append(tfeatures.StateFeatExtractor(featId))
            
    extractor = tfeatures.TransFeatureExtractor(stateExtractors)
    return extractor

def buildTransParser(opts, dummyBuilder, reprBuilder):
    tsystem, anoracle = _buildOracleAndTransSystem(opts)
    transExtractor = _buildFeatureExtractor(opts.features)
    
    featIds = transExtractor.getFeatIds()
    transNetwork = nparser.network.ParserNetwork(opts.mlpHiddenDim, opts.nonLinFun, featIds, opts.trainMargin)
    featBuilder = nparser.features.FeatReprBuilder(transExtractor, { }, dummyBuilder, transNetwork, opts.parseLayer)
    
    parsingTask = task.NNTransParsingTask(tsystem, anoracle, transNetwork, featBuilder)
    return parsingTask
