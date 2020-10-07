'''
Created on 23.08.2017

@author: falensaa
'''

from tools import utils

def addReprCmdArguments(argParser):
    # word representations
    reprArgs = argParser.add_argument_group('Token representations')
    reprArgs.add_argument("--contextRepr", help="context representation", choices=[ "bilstm", "concat" ], required=False, default="bilstm")
    reprArgs.add_argument("--representations", help="which representations to use, select a subset of [word|pos|char|morph|eword|elmo], use ',' separator", required=False, default="word,pos,char")
    
    # word representations
    reprArgs.add_argument("--wordDim", help="word dimensionality", required=False, default=100, type=int)
    reprArgs.add_argument("--wordDropout", help="word dropout", required=False, default="0.25")

    # char representations
    reprArgs.add_argument("--charDim", help="char dimensionality", required=False, default=24, type=int)
    reprArgs.add_argument("--charLstmDim", help="char lstm dimensionality", required=False, default=100, type=int)
    reprArgs.add_argument("--charDropout", help="char dropout", required=False, default="None") #0.25
    reprArgs.add_argument("--charLstmDropout", help="char lstm dropout", required=False, default="0.33")
    
    # other dimensionalities
    reprArgs.add_argument("--posDim", help="pos dimensionality", required=False, default=20, type=int)
    reprArgs.add_argument("--morphDim", help="morph dimensionality", required=False, default=50, type=int)
    
    # bilstm features
    bilstmArgs = argParser.add_argument_group('BiLSTM')
    bilstmArgs.add_argument("--lstmDim", help="lstmDim", required=False, default=125, type=int)
    bilstmArgs.add_argument("--lstmDropout", help="lstm Dropout", required=False, default="0.33")
    bilstmArgs.add_argument("--lstmLayers", help="nr of layers", required=False, type=int, default=2)
    
    # elmo model
    elmoArgs = argParser.add_argument_group('ELMo model')
    elmoArgs.add_argument("--elmoFile", help="HDF5 file that contains ELMo layers for sentences", required=False, default=None)
    elmoArgs.add_argument("--elmoGamma", help="Gamma factor to tune ELMo", required=False, type=float, default=1.0)
    elmoArgs.add_argument("--elmoLearnGamma", help="Learn the gamma factor for ELMo", required=False, type=str, default="True")
    
    # external embeddings
    extArgs = argParser.add_argument_group('External embeddings')
    extArgs.add_argument("--embFile", help="file with external embeddings", required=False)
    extArgs.add_argument("--embUpdate", help="update external embeddings", choices=[ "True", "False" ], required=False, default="True")
    extArgs.add_argument("--embDim", help="external embeddings dimensionality", required=False, default=300, type=int)
    extArgs.add_argument("--embLowercased", help="are embeddings lowercased", required=False, default="False", type=str)
    
    # stack settings
    stackArgs = argParser.add_argument_group('Stacking settings')
    stackArgs.add_argument("--stackRepr", help="which representations to use, select a subset of [head|lbl|stag], use ',' separator", required=False)
    stackArgs.add_argument("--stagDim", help="stag dimensionality", required=False, default=30, type=int)
    stackArgs.add_argument("--stackLblDim", help="stacked label dimensionality", required=False, default=30, type=int)
    stackArgs.add_argument("--stackHeadDim", help="stacked head dimensionality", required=False, default=100, type=int)
    
    
def fillReprOptions(args, opts):
    ### representations
    opts.addOpt("contextRepr", args.contextRepr)
    opts.addOpt("representations", args.representations.split(","))
    
    ### words
    if "word" in opts.representations or "eword" in opts.representations:
        opts.addOpt("wordDim", args.wordDim)
        opts.addOpt("wordDropout", float(args.wordDropout) if args.wordDropout != "None" else None)
    
    ### chars
    if "char" in opts.representations:
        opts.addOpt("charDim", args.charDim)
        opts.addOpt("charLstmDim", args.charLstmDim)
        opts.addOpt("charDropout", float(args.charDropout) if args.charDropout != "None" else None)
        opts.addOpt("charLstmDropout", float(args.charLstmDropout) if args.charLstmDropout != "None" else None)
        
    
    ### pos
    if "pos" in opts.representations:
        opts.addOpt("posDim", args.posDim)
    
    ### morph    
    if "morph" in opts.representations: 
        opts.addOpt("morphDim", args.morphDim)
        
    ### elmo
    if "elmo" in opts.representations:
        if args.elmoFile is not None:
            opts.addOpt("elmoFile", args.elmoFile, override=True)
            
        opts.addOpt("elmoGamma", args.elmoGamma)
        opts.addOpt("elmoLearnGamma", utils.parseBoolean(args.elmoLearnGamma))
    
    ### eword    
    if "eword" in opts.representations:
        if args.embFile is not None:
            opts.addOpt("embFile", args.embFile, override=True)
            
        opts.addOpt("embUpdate", utils.parseBoolean(args.embUpdate))
        opts.addOpt("embDim", args.embDim)
        opts.addOpt("embLowercased", utils.parseBoolean(args.embLowercased))
        
    opts.addOpt("stackRepr", args.stackRepr.split(",") if args.stackRepr else None)
    
    ### stacking
    if args.stackRepr:
        ### stag
        if "stag" in opts.stackRepr:
            opts.addOpt("stagDim", args.stagDim)
        
        if "lbl" in opts.stackRepr:
            opts.addOpt("stackLblDim", args.stackLblDim)
            
        if "head" in opts.stackRepr:
            opts.addOpt("stackHeadDim", args.stackHeadDim)
    
    ### lstms
    if args.contextRepr == "bilstm":
        opts.addOpt("lstmDim", args.lstmDim)
        opts.addOpt("lstmDropout", float(args.lstmDropout) if args.lstmDropout != "None" else None)
        opts.addOpt("lstmLayers", args.lstmLayers)
    
    return opts
