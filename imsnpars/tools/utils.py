'''
Created on 03-07-2014

@author: falensaa
'''
   
import logging
import re
import sys
import pickle
import gzip

from tools import datatypes

############################### options #########################################################

class NParserOptions(object):
    def __init__(self):
        pass
    
    def copy(self):
        result = NParserOptions()
        result.__dict__ = self.__dict__.copy()
        return result
    
    def logOptions(self):
        logging.debug("Params: %s" % self.__dict__)
        
    def save(self, filename):
        with open(filename, 'wb') as outFile:
            pickle.dump(self.__dict__, outFile)

    def load(self, filename):
        with open(filename, 'rb') as inFile:
            stored = pickle.load(inFile)
            self.__dict__.update(stored)
            
    def addOpt(self, name, value, override = False):
        if name in self.__dict__ and not override:
            return
        
        self.__dict__[name] = value
        
        
def parseBoolean(v):
    return v != None and v.lower() in ("yes", "true", "t", "1")

############################### datatypes #########################################################

class ConLLToken():
    def __init__(self, tokId, orth, lemma, pos, langPos, morph, headId, dep, norm):
        self.tokId = tokId
        self.orth = orth
        self.lemma = lemma
        self.pos = pos
        self.langPos = langPos
        self.morph = morph
        self.headId = headId
        self.dep = dep
        self.norm = norm

    def __str__(self):
        return "\t".join(self.collectParts())
    
    def collectParts(self):
        result = [ self.getId(), encode(self.orth), self.lemma, self.pos, self.langPos, self.morph, self.__niceId(self.headId), self.__niceVal(self.dep)]
        return result
    
    def getId(self):
        return str(self.tokId)
    
    def __niceId(self, aId):
        if aId == None:
            return "_"
        else:
            return str(aId)

    def __niceVal(self, aVal):
        if aVal == None:
            return "_"
        else:
            return aVal
        
    def getTokPos(self):
        return self.tokId - 1

    def getHeadPos(self):
        return self.headId - 1
    
    def setHeadPos(self, headPos):
        if headPos == None:
            self.headId = None
        else:
            self.headId = headPos + 1
    
    def copy(self):
        return ConLLToken(self.tokId, self.orth, self.lemma, self.pos, self.langPos, self.morph, self.headId, self.dep, self.norm)
    
    def getNorm(self):
        if self.norm == None:
            return self.orth
        else:
            return self.norm
        
############################### readers #########################################################

class LazySentenceReader():
    def __init__(self, f, tokenBuilder, normalizer = None):
        self.__f = f
        self.__el = None
        self.__tokBuilder = tokenBuilder
        self.__normalizer = normalizer
        
    def top(self):
        return self.__el
    
    def next(self):
        sentence = [ ]
        line = self.__f.readline().strip()
        while line:
            tok = self.__tokBuilder(line, self.__normalizer)
            if tok:
                sentence.append(tok)
            line = self.__f.readline().strip()
            
        self.__el = sentence
        return sentence
    
class LazyTokenReader(object):
    def __init__(self, f, tokenBuilder):
        self.reader = LazySentenceReader(f, tokenBuilder)
        self.__pos = None
        
    def top(self):
        return self.reader.top()[self.__pos]
    
    def next(self):
        if self.reader.top() == None or self.__pos >= len(self.reader.top()):
            self.reader.next()
            self.__pos = 0
        
        if not self.reader.top():
            return None
        
        result = self.reader.top()[self.__pos]
        self.__pos += 1
        return result
    
    def getSent(self):
        return self.reader.top()


def readSentences(filename, builder, normalizer = None):
    result = [ ]
    reader = LazySentenceReader(open(filename), builder, normalizer)            
    sent = reader.next()
    while sent:
        result.append(sent)
        sent = reader.next()
    return result
        
def buildFormatReader(fformat):
    # format reader
    if fformat == "conll06":
        tokenBuilder = buildTokenFromConLL06
    elif fformat == "conllu":
        tokenBuilder = buildTokenFromConLLU
    else:
        logging.error("Unknown format: %s" % fformat)
        exit()
        
    return tokenBuilder

def buildTokenFromConLL06(line, normalizer = None):
    if type(line) != type([]):
        parts = line.split("\t")
    else:
        parts = line
    
    if len(parts) < 10:
        raise Exception("Too short line: %s" % line)
        
    return ConLLToken(tokId=int(parts[0]),
                      orth = decode(parts[1]),
                      lemma = parts[2],
                      pos = parts[3],
                      langPos = parts[4],
                      morph = parts[5],
                      headId = int(parts[6]) if parts[6] != '_' else None,
                      dep = parts[7],
                      norm = normalizer.norm(decode(parts[1])) if normalizer != None else None)

def buildTokenFromConLLU(line, normalizer = None):
    if line.strip()[0] == "#":
        return None
    
    if type(line) != type([]):
        parts = line.split("\t")
    else:
        parts = line
    
    if len(parts) < 10:
        raise Exception("Too short line: %s" % line)
    
    # when IDs come from ConLL09
    if "_" in parts[0]:
        tokId = parts[0].split("_")[1]
    else:
        tokId = parts[0]
        
    return ConLLToken(tokId=int(tokId),
                      orth = decode(parts[1]),
                      lemma = parts[2],
                      pos = parts[3],
                      langPos = parts[4],
                      morph = parts[5],
                      headId = int(parts[6]) if parts[6] != '_' else None,
                      dep = parts[7],
                      norm = normalizer.norm(decode(parts[1])) if normalizer != None else None)
    

############################### writers #########################################################

def buildConLLUFromToken(token):
    parts = token.collectParts()
    parts.append("_")
    parts.append("_")
    return "\t".join(parts)
    
class LazySentenceWriter(object):
    def __init__(self, f, tokenToStr = buildConLLUFromToken):
        self.__f = f
        self.__tokenToStr = tokenToStr
        
    def processTree(self, sentence, tree):
        for pos, tok in enumerate(sentence):
            ntok = tok.copy()
            ntok.setHeadPos(tree.getHead(pos))
            ntok.dep = tree.getLabel(pos)
            self.__f.write(self.__tokenToStr(ntok) + "\n")
            
        self.__f.write("\n")
        
    def processPosTags(self, sentence, tags):
        for tok, tag in zip(sentence, tags):
            ntok = tok.copy()
            ntok.pos = tag
            self.__f.write(self.__tokenToStr(ntok) + "\n")
            
        self.__f.write("\n")
    
    def processSentence(self, sentence):
        for tok in sentence:
            self.__f.write(self.__tokenToStr(tok) + "\n")
            
        self.__f.write("\n")
        

############################### normalizers #########################################################

class Normalizer(object):
    NR = '##__NUM__##'
    
    def __init__(self, normalizeNumbers, lowercase):
        self.__numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+")
        
        self.__normalizers = [ ]
        
        if normalizeNumbers:
            self.__normalizers.append(self.__normNrs)
            
        if lowercase:
            self.__normalizers.append(self.__normLower)
        
    def norm(self, txt):
        for nFun in self.__normalizers:
            txt = nFun(txt)
        return txt
    
    def __normNrs(self, txt):
        return self.NR if self.__numberRegex.match(txt) else txt
    
    def __normLower(self, txt):
        return txt.lower()


def buildNormalizer(normalize):
    if normalize:
        return Normalizer(normalizeNumbers = True, lowercase = True)
    else:
        return None
    
############################### utils #########################################################

def first(filterFun, elems):
    for elem in elems:
        if filterFun(elem):
            return elem
        
    return None

def decode(s):
    # checks for python version
    if sys.version_info.major == 2:
        return s.decode("utf-8")
    elif type(s) == bytes:
        return s.decode("utf-8")
    else:
        return s
    
def encode(s):
    # checks for python version
    if sys.version_info.major == 2:
        return s.encode("utf-8")
    else:
        return s
    
def smartOpen(f, mode = 'r'):
    if f.endswith('.gz'):
        return gzip.open(f, mode)
    
    return open(f, mode)

def filterNonProjective(sentences):
    result = [ ]
    filtered = 0
    for sent in sentences:
        tree = datatypes.sentence2Tree(sent)
        if tree.isProjective():
            result.append(sent)
        else:
            filtered += 1
            
    return result




