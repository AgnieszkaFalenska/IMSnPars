'''
Created on 23.08.2017

@author: falensaa
'''

import logging

import nparser.network
import nparser.features
import nparser.trans.builder as tbuilder
import nparser.trans.labeler as tlabeler
import nparser.graph.builder as gbuilder
import nparser.graph.features as gfeatures
import nparser.labels.task as ltask

def buildGraphLabeler(opts, dummyBuilder, reprBuilder):
    reprDim = reprBuilder.getDim()
    
    # parse features
    tokExtractors, featBuilders = gbuilder.buildGraphFeatureExtractors(opts.graphLabelerFeats, reprDim)
    extractor = gfeatures.GraphFeatureExtractor(tokExtractors)
    
    featIds = extractor.getFeatIds() + [ feat.getFeatId() for feat in featBuilders.values() ]
    
    # all network's parameters
    lblNetwork = nparser.network.ParserNetwork(opts.mlpHiddenDim, opts.nonLinFun, featIds)
    featBuilder = nparser.features.FeatReprBuilder(extractor, featBuilders, dummyBuilder, lblNetwork, opts.lblLayer)
    labelerTask = ltask.LabelerGraphTask(featBuilder, lblNetwork, opts.lblLayer)
    return labelerTask
    