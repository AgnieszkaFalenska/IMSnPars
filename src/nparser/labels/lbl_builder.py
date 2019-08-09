'''
Created on 23.08.2017

@author: falensaa
'''

import logging

from nparser.graph.gp_builder import buildGraphFeatureExtractors
from nparser.labels import lbl_task
from nparser.pars_network import ParserNetwork
from nparser.graph.gp_features import GraphFeatureExtractor
from nparser.pars_features import FeatReprBuilder
from nparser.trans import tp_builder
from nparser.trans.tp_labeler import TransSystemLabeler

def buildGraphLabeler(options, dummyBuilder, reprBuilder):
    reprDim = reprBuilder.getDim()
    
    # parse features
    tokExtractors, featBuilders = buildGraphFeatureExtractors(options.graphLabelerFeats, reprDim)
    extractor = GraphFeatureExtractor(tokExtractors)
    
    featIds = extractor.getFeatIds() + [ feat.getFeatId() for feat in featBuilders.values() ]
    
    # all network's parameters
    lblNetwork = ParserNetwork(options.mlpHiddenDim, options.nonLinFun, featIds)
    featBuilder = FeatReprBuilder(extractor, featBuilders, dummyBuilder, lblNetwork, options.lblLayer)
    labelerTask = lbl_task.LabelerGraphTask(featBuilder, lblNetwork, options.lblLayer)
    return labelerTask

def _buildTransLblSystem(options):
    tsystem = tp_builder._buildTransSystem(options)
    transLabeler = TransSystemLabeler(tsystem)
    anoracle = tp_builder._buildOracle(options, tsystem, transLabeler)
    noLblOracle = tp_builder._buildOracle(options, tsystem, None) 
         
    logging.info("trans-mtl: Trans system used: %s" % type(tsystem))
    logging.info("trans-mtl: Oracle used: %s" % type(anoracle))
    logging.info("trans-mtl: Trans-labeler used: %s" % type(transLabeler))
        
    return transLabeler, anoracle, noLblOracle

def buildTransLabeler(options, dummyBuilder, reprBuilder):
    tsystem, anoracle, noLblOracle = _buildTransLblSystem(options)
    transExtractor = tp_builder._buildFeatureExtractor(options.features)
    
    featIds = transExtractor.getFeatIds()
    
    transNetwork = ParserNetwork(options.mlpHiddenDim, options.nonLinFun, featIds)
    featBuilder = FeatReprBuilder(transExtractor, { }, dummyBuilder, transNetwork, options.parseLayer)
    lblTask = lbl_task.LabelerTransTask(tsystem, anoracle, noLblOracle, transNetwork, featBuilder)

    return lblTask
    