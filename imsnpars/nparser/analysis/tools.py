'''
Created on 10.02.2019

@author: falensaa
'''

import abc
import numpy as np
import dynet
import logging

from tools import datatypes, utils

def _getGradient(embed):
    return np.linalg.norm(embed.gradient(), ord=2)

class IMSnParsAnalysisTool(object):
    __metaclass__ = abc.ABCMeta
    
    
    @abc.abstractmethod
    def runAnalysis(self, sentences, output):
        pass
    
    
class SurfGradientAnalysis(IMSnParsAnalysisTool):
    
    def __init__(self, reprLstmBuilder, lstmLayer):
        self.__reprLstmBuilder = reprLstmBuilder
        self.__lstmLayer = lstmLayer
        
    def __addToResult(self, name, pos, tokId, grad, sentGrad, result):
        if name not in result:
            result[name] = { }
            
        if pos not in result[name]:
            result[name][pos] = [ ]
        
        result[name][pos].append(((tokId, grad, sentGrad)))
    
    def __dumpResult(self, sentId, result, out):
        outFormat = "[%s] sent=%i pos=%i grad=%s full=%s\n"
        
        toPrint = [ ]
        names = sorted(result.keys())
        for name in names:
            poss = sorted(result[name].keys())
            for pos in poss:
                values = result[name][pos]
                for tokId, grad, sentGrad in values:
                    toPrint.append((sentId, tokId, name, pos, str(grad), str(sentGrad)))
                        
        toPrint.sort()
        for sentId, tokId, name, pos, grad, sentGrad in toPrint:
            out.write(outFormat % (name, sentId, pos, grad, sentGrad))
                
    def runAnalysis(self, sentences, output):
        out = open(output, "w")
        
        for sentId, sent in enumerate(sentences):
            dynet.renew_cg()
            
            result = { }
            
            instance = self.__reprLstmBuilder.buildInstance(sent)
            embeds = self.__reprLstmBuilder.prepareEmbeddingsForLSTM(instance, isTraining=False)
            vectors = self.__reprLstmBuilder.applyLSTMToEmbeds(embeds, isTraining = False)
            
            correctTree = datatypes.sentence2Tree(sent)
            for pos in range(len(sent)):
                # prepare info about the tree
                children = sorted(correctTree.getChildren(pos))
                headPos = correctTree.getHead(pos)
                
                if headPos != -1:
                    grandPos = correctTree.getHead(headPos)
                else:
                    grandPos = None
                
                sibl = [ ]
                for child in correctTree.getChildren(headPos):
                    if child != pos:
                        sibl.append(child)
                
                
                # back-propagate -- all positions
                dynet.sum_elems(vectors.wordsV[pos][self.__lstmLayer - 1]).backward()
                wordGradients = [ _getGradient(embed) for embed in embeds.wordsV ]
                sentGrad = sum(wordGradients)
                
                # write for every position
                for cPos in range(len(sent)):
                    grad = wordGradients[cPos]
                    if cPos == headPos:
                        self.__addToResult("Head", pos - cPos, pos, grad, sentGrad, result)
                    else:
                        self.__addToResult("Other", pos - cPos, pos, grad, sentGrad, result)
    
            self.__dumpResult(sentId, result, out)
