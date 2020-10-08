# IMSnPars at IWPT2020

IMS Neural Dependency Parser is a re-implementation of the transition- and graph-based parsers described in [Simple and Accurate Dependency Parsing
Using Bidirectional LSTM Feature Representations](https://aclweb.org/anthology/Q16-1023)

This version of the parser was developed for the paper [Integrating Graph-Based and Transition-Based Dependency Parsers in the Deep Contextualized Era](https://www.aclweb.org/anthology/2020.iwpt-1.4.pdf). The branch contains all the specific changes and analysis tools.

```
@inproceedings{falenska-etal-2020-integrating,
    title = "Integrating Graph-Based and Transition-Based Dependency Parsers in the Deep Contextualized Era",
    author = {Falenska, Agnieszka  and Bj{\"o}rkelund, Anders  and Kuhn, Jonas},
    booktitle = "Proceedings of the 16th International Conference on Parsing Technologies and the IWPT 2020 Shared Task on Parsing into Enhanced Universal Dependencies",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.iwpt-1.4",
    doi = "10.18653/v1/2020.iwpt-1.4",
    pages = "25--39",
}
```

## Required software

> Python 3.7

> [Dynet 2.0](http://dynet.io/)

> [NetworkX package](https://networkx.github.io/)

## Usage

### Transition-based parser

Training a new model:
```
python3 imsnpars/main.py --parser TRANS --train [train_file] --save [model_file]
```

Loading and predicting with the trained model:
```
python3 imsnpars/main.py --parser TRANS --model [model_file] --test  [test_file] --output [output_file]
```

The parser supports many other options. All of them can be seen after running:
```
python3 imsnpars/main.py --parser TRANS --help
```

### Graph-based parser

Training a new model:
```
python3 imsnpars/main.py --parser GRAPH --train [train_file] --save [model_file]
```

Loading and predicting with the trained model:
```
python3 imsnpars/main.py --parser GRAPH --model [model_file] --test  [test_file] --output [output_file]
```

The parser supports many other options. All of them can be seen after running:
```
python3 imsnpars/main.py --parser GRAPH --help
```

### Multi-task parser

Training a new model:
```
python3 imsnpars/main.py --parser MTL --train [train_file] --save [model_file] --firstTask [GRAPH|TRANS] --secondTask [GRAPH|TRANS] --t1Labeler [trans|graph|graph-mtl|None] --t2Labeler [trans|graph|graph-mtl|None] 
```

Loading and predicting with the first task:
```
python3 imsnpars/main.py --parser MTL --model [model_file] --test  [test_file] --output [output_file]
```

Loading and predicting with the second task:
```
python3 imsnpars/main.py --parser MTL --model [model_file] --test  [test_file] --t2Output [output_file]
```

The parser supports many other options. All of them can be seen after running:
```
python3 imsnpars/main.py --parser MTL --help
```

## Tests

*IMSnPars* comes with six testing scripts to check if everything works fine:
1. systests/test_trans_parser.sh -- trains a new transition-based parser on small fake data and uses this model for prediction
2. systests/test_graph_parser.sh -- trains a new graph-based parser on small fake data and uses this model for prediction
3. systests/test_fasttext_parser.sh -- trains a new parser using external embeddings
4. systests/test_elmo_parser.sh -- trains a new parser using ELMo representations
5. systests/test_all_trans_parsers.sh -- trains multiple transition-based models with different sets of options
6. systests/test_all_graph_parsers.sh -- trains multiple graph-based models with different sets of options

We recommend running the four first scripts before using *IMSnPars* for other purposes (all tests take less than a minute). All of the scripts should end with an information that everything went fine. For the first two tests: transition-based parser achieves LAS=64.61 on the fake data and the graph-based one LAS=66.47.

### Branch Specific Tests

The *iwpt2020* branch comes with three additional testing scripts to check if everything works fine:
1. systests/test_mtl_parser.sh -- trains a new parser on small fake data and uses it for prediction. The parser is built from two models -- graph-based and transition-based -- that share intermediate representations.
2. systests/test_stag_parser.sh -- trains two parsers -- graph-based and transition-based -- with additional supertag features
3. systests/test_all_mtl_parsers.sh -- trains multiple MTL models with different sets of options
4. systests/test_surf_analysis.sh --  runs surface analysis with a small fake model

## Analysis tools -- BiLSTM representation analysis

To collect gradient values for words with different surface distance to the BiLSTM representation (see Figure 4 in the paper) run:

``` 
python3 imsnpars/analyze.py --parser [GRAPH|TRANS|MTL] --analyze SURF --model [model_file] --test [test_file] --output [output_file]
```

