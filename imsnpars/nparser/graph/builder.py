'''
Created on 23.08.2017

@author: falensaa
'''

import logging
import sys

import nparser.features
import nparser.network
import nparser.graph.features as gfeatures
from nparser.graph import task, decoder
from nparser.graph.mst import cle
from nparser.labels import task as ltask

def buildMSTDecoder(opts, featBuilder):
    if opts.mst == "CLE":
        mstAlg = cle.ChuLiuEdmonds()
        decod = decoder.FirstOrderDecoder(featBuilder)
    else:
        logging.error("Unknown algorithm: %s" % opts.mst)
        sys.exit()
    
    logging.info("Graph system used: %s" % type(mstAlg))
    logging.info("Decoder used: %s" % type(decod))
    return mstAlg, decod

def buildGraphFeatureExtractors(featuresD, reprDim):
    featIds = { ("h", "0"): gfeatures.FeatId.HEAD,
                ("d", "0"): gfeatures.FeatId.DEP,
                ("h", "1"): gfeatures.FeatId.HEAD_P_1,
                ("h", "2"): gfeatures.FeatId.HEAD_P_2,
                ("d", "1"): gfeatures.FeatId.DEP_P_1,
                ("d", "2"): gfeatures.FeatId.DEP_P_2,
                ("h", "-1"): gfeatures.FeatId.HEAD_M_1,
                ("h", "-2"): gfeatures.FeatId.HEAD_M_2,
                ("d", "-1"): gfeatures.FeatId.DEP_M_1,
                ("d", "-2"): gfeatures.FeatId.DEP_M_2,
                ("dist", "0") : gfeatures.FeatId.DIST }
    
    mainFeatIds = {"h": gfeatures.FeatId.HEAD,
                   "d": gfeatures.FeatId.DEP }
    
    featureExtractors = { }
    featureBuilders = { }
    
    for feat in featuresD:
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
        if featId == gfeatures.FeatId.DIST:
            featureBuilders[featId] = gfeatures.DistFeatureBuilder(reprDim)
        else:
            mainFeature = mainFeatIds[name]
            if mainFeature not in featureExtractors:
                featureExtractors[mainFeature] = gfeatures.TokenFeatExtractor()
                
            featureExtractors[mainFeature].addShift(featId, int(shift))
       
    return featureExtractors, featureBuilders
    
        
def buildGraphParser(opts, dummyBuilder, reprBuilder):
    reprDim = reprBuilder.getDim()
    tokExtractors, featBuilders = buildGraphFeatureExtractors(opts.features, reprDim)
    extractor = gfeatures.GraphFeatureExtractor(tokExtractors)
    
    featIds = extractor.getFeatIds() + [ feat.getFeatId() for feat in featBuilders.values() ]
    network = nparser.network.ParserNetwork(opts.mlpHiddenDim, opts.nonLinFun, featIds)
    featBuilder = nparser.features.FeatReprBuilder(extractor, featBuilders, dummyBuilder, network, opts.parseLayer)
    mstAlg, decod = buildMSTDecoder(opts, featBuilder)
    
    if opts.labeler == "graph":
        lblDict = ltask.LblTagDict()
        parsingTask = task.NNGraphParsingTaskWithLbl(mstAlg, featBuilder, decod, network, opts.augment, lblDict)
    else:
        parsingTask = task.NNGraphParsingTask(mstAlg, featBuilder, decod, network, opts.augment)
    
    return parsingTask
