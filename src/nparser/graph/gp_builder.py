'''
Created on 23.08.2017

@author: falensaa
'''

import logging
import sys

from nparser.graph import mst, cle
from nparser.graph.gp_features import FeatId, DistFeatureBuilder, TokenFeatExtractor, GraphFeatureExtractor
from nparser.graph.gp_task import NNGraphParsingTask, NNGraphParsingTaskWithLbl
from nparser.graph.gp_decoder import FirstOrderDecoder
from nparser.pars_network import ParserNetwork
from nparser.pars_features import FeatReprBuilder
from nparser.labels.lbl_task import LblTagDict

def buildMSTDecoder(options, featBuilder):
    if options.mst == "CLE":
        mstAlg = cle.ChuLiuEdmonds()
        decoder = FirstOrderDecoder(featBuilder)
    else:
        logging.error("Unknown algorithm: %s" % options.mst)
        sys.exit()
    
    logging.info("Graph system used: %s" % type(mstAlg))
    logging.info("Decoder used: %s" % type(decoder))
    return mstAlg, decoder

def buildGraphFeatureExtractors(features, reprDim):
    featIds = { ("h", "0"): FeatId.HEAD,
                ("d", "0"): FeatId.DEP,
                ("h", "1"): FeatId.HEAD_P_1,
                ("h", "2"): FeatId.HEAD_P_2,
                ("d", "1"): FeatId.DEP_P_1,
                ("d", "2"): FeatId.DEP_P_2,
                ("h", "-1"): FeatId.HEAD_M_1,
                ("h", "-2"): FeatId.HEAD_M_2,
                ("d", "-1"): FeatId.DEP_M_1,
                ("d", "-2"): FeatId.DEP_M_2,
                ("dist", "0") : FeatId.DIST }
    
    mainFeatIds = {"h": FeatId.HEAD,
                   "d": FeatId.DEP }
    
    featureExtractors = { }
    featureBuilders = { }
    
    for feat in features:
        if "+" in feat:
            name, shift = feat.split("+")
        elif "-" in feat:
            name, shift = feat.split("-")
            shift = "-" + shift
        else:
            name, shift = feat, "0"
            
        featId = featIds.get((name, shift))
        if featId == None:
            logging.error("Unknown token id: %s" % feat)
            sys.exit()
        
        # for now there is only one builder -- distance
        if featId == FeatId.DIST:
            featureBuilders[featId] = DistFeatureBuilder(reprDim)
        else:
            mainFeature = mainFeatIds[name]
            if mainFeature not in featureExtractors:
                featureExtractors[mainFeature] = TokenFeatExtractor()
                
            featureExtractors[mainFeature].addShift(featId, int(shift))
       
    return featureExtractors, featureBuilders
    
        
def buildGraphParser(options, dummyBuilder, reprBuilder):
    reprDim = reprBuilder.getDim()
    tokExtractors, featBuilders = buildGraphFeatureExtractors(options.features, reprDim)
    extractor = GraphFeatureExtractor(tokExtractors)
    
    featIds = extractor.getFeatIds() + [ feat.getFeatId() for feat in featBuilders.values() ]
    network = ParserNetwork(options.mlpHiddenDim, options.nonLinFun, featIds)
    featBuilder = FeatReprBuilder(extractor, featBuilders, dummyBuilder, network, options.parseLayer)
    mstAlg, decoder = buildMSTDecoder(options, featBuilder)
    
    if options.labeler == "graph":
        lblDict = LblTagDict()
        parsingTask = NNGraphParsingTaskWithLbl(mstAlg, featBuilder, decoder, network, options.augment, lblDict)
    else:
        parsingTask = NNGraphParsingTask(mstAlg, featBuilder, decoder, network, options.augment)
    
    return parsingTask
