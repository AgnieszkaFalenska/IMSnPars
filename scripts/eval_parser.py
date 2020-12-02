'''
Created on 14.11.2017

@author: falensaa
'''

def evalFile(corrReader, lreader, accName, evalFullLabels):
    lazyEval = evaluator.LazyTreeEvaluator(evalFullLabels)
    
    corrSent = corrReader.next()
    predSent = lreader.next()
    
    while corrSent:
        assert len(corrSent) == len(predSent)
        assert all([t1.orth == t2.orth for (t1, t2) in zip(corrSent, predSent)])
        
        predTree = datatypes.sentence2Tree(predSent)
        lazyEval.processTree(corrSent, predTree)
        
        corrSent = corrReader.next()
        predSent = lreader.next()
    
    assert not corrReader.next()
    assert not lreader.next()
        
    result = lazyEval.calcAcc(accName)

    if info == "all":
        print("%.2f  %.2f" % result)
    else:
        print("%.2f" % result)
        
if __name__ == '__main__':
    import os, sys

    thisFile = os.path.realpath(__file__)
    thisPath = os.path.dirname(thisFile)
    parentPath = os.path.dirname(thisPath)
    sys.path.append(parentPath + os.path.sep + "imsnpars")
    
    from tools import utils, datatypes, evaluator
    
    args = sys.argv[1:]

    if len(args) < 2:
        print('Error: Missing argument!', file=sys.stderr)
        print('eval_parser goldFile conlluFile', file=sys.stderr)
        sys.exit()
        
    builder=utils.buildTokenFromConLLU
    goldFile = utils.LazySentenceReader(open(args[0], "r"), builder)
    conllFile = utils.LazySentenceReader(open(args[1], "r"), builder)
    
    if len(args) > 2:
        info=args[2]
    else:
        info="all"

    if len(args) > 3:
        fullLabels = args[3] in [ "True", "1", "yes", "true" ]
    else:
        fullLabels = False
    
    evalFile(goldFile, conllFile, info, fullLabels)
