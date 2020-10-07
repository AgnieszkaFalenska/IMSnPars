'''
Created on 2 lip 2014

@author: agnieszka
'''

import sys

class SuperTagger(object):
    '''Model creates tags from heads and directions of dependents'''
    
    def __init__(self):
        pass
    
    def annotate(self, sentence, parsedSent):
        deps = { }    
        for parsedTok in parsedSent:
            stackHead = parsedTok.getHeadPos()
            if stackHead not in deps:
                deps[stackHead] = [ ]
            
            depRel = self.__getDepRel(parsedTok.getTokPos(), stackHead)
            deps[stackHead].append(depRel)
            
            
        for tok, parsedTok in zip(sentence, parsedSent):
            assert tok.orth == parsedTok.orth
            
            superTag = self.__mainSuperTag(parsedTok)
            
            if parsedTok.getTokPos() in deps:
                superTag += "+" + "_".join(sorted(list(set(deps[parsedTok.getTokPos()]))))
                
            
            stackHead = parsedTok.headId
            stackDep = parsedTok.dep
            
            tok.addSuperTag(superTag)
            tok.addStackInfo(stackHead, stackDep)
        
    def __getHeadRel(self, tokPos, headPos):
        if headPos > tokPos:
            return "R"
        elif headPos < tokPos:
            return "L"
        else:
            print("Problem, the same tokPos and headPos:", tokPos, headPos)
            return None

    def __getDepRel(self, tokPos, headPos):
        if headPos < tokPos:
            return "R"
        elif headPos > tokPos:
            return "L"
        else:
            print("Problem, the same tokPos and headPos:", tokPos, headPos)
            return None
    
    def __mainSuperTag(self, parsedToken):
        result = parsedToken.dep
        result += "/" + self.__getHeadRel(parsedToken.getTokPos(), parsedToken.getHeadPos())
        return result
    
    
def convertTreesToStags(sentences, parsedReader, writer):
    tagger = SuperTagger()
    
    parsedSent = parsedReader.next()
    dataId = 0
    while parsedSent:
        dataSent = sentences[dataId]
        assert len(parsedSent) == len(dataSent)
        
        tagger.annotate(dataSent, parsedSent)
        writer.processSentence(dataSent)
        
        parsedSent = parsedReader.next()
        dataId += 1
        
    assert dataId == len(sentences)
        
if __name__ == "__main__":
    import os

    thisFile = os.path.realpath(__file__)
    thisPath = os.path.dirname(thisFile)
    parentPath = os.path.dirname(thisPath)
    sys.path.append(parentPath)

    from tools import utils
    
    inFile = sys.argv[1]
    parsedFile = sys.argv[2]
    outFile = sys.argv[3]
    
    tokenBuilder = utils.buildFormatReader("conllu")
    normalizer =  utils.buildNormalizer(True)
    
    sentences = utils.readSentences(inFile, tokenBuilder, normalizer)
    parsedReader = utils.LazySentenceReader(open(parsedFile), tokenBuilder, normalizer)
    stagWriter = utils.LazySentenceWriter(open(outFile, "w"), tokenToStr=utils.buildConLLUStackFromToken)
    
    convertTreesToStags(sentences, parsedReader, stagWriter)