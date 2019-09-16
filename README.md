# IMSnPars at ACL2019

IMS Neural Dependency Parser is a re-implementation of the transition- and graph-based parsers described in [Simple and Accurate Dependency Parsing
Using Bidirectional LSTM Feature Representations](https://aclweb.org/anthology/Q16-1023)

The parser was developed for the paper [The (Non-)Utility of Structural Features in BiLSTM-based
Dependency Parsers](https://www.aclweb.org/anthology/P19-1012).

```
@inproceedings{falenska-kuhn-2019-non,
    title = "The (Non-)Utility of Structural Features in {B}i{LSTM}-based Dependency Parsers",
    author = "Falenska, Agnieszka  and Kuhn, Jonas",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1012",
    doi = "10.18653/v1/P19-1012",
    pages = "117--128",
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

### Tests

*IMSnPars* comes with four testing scripts to check if everything works fine:
1. systests/test_trans_parser.sh -- trains a new transition-based parser on small fake data and uses this model for prediction
2. systests/test_graph_parser.sh -- trains a new graph-based parser on small fake data and uses this model for prediction
3. systests/test_all_trans_parsers.sh -- trains multiple transition-based models with different sets of options
4. systests/test_all_graph_parsers.sh -- trains multiple graph-based models with different sets of options

We recommend running the two first scripts before using *IMSnPars* for other purposes (both tests take less than a minute). Both of the scripts should end with an information that everything went fine. Transition-based parser achieves LAS=64.61 on the fake data and the graph-based one LAS=66.47.

## Analysis tools

### BiLSTM representation analysis

To collect gradient values for words with different surface distance to the BiLSTM representation (see Figure 4 in the paper) run:

``` 
python3 imsnpars/analyze.py --parser [GRAPH,TRANS] --analyze SURF --model [model_file] --test [test_file] --output [output_file]
```

### Configuration/arc decision analysis

To collect gradient values for words at different structural positions (see Figure 5 in the paper) run:

``` 
python3 imsnpars/analyze.py --parser [GRAPH,TRANS] --analyze FEAT --model [model_file] --test [test_file] --output [output_file]
```

### Drop analysis

To train/run a parser without a particular structural position (see Figure 6 in the paper) run:

``` 
# training
imsnpars/main.py --parser [GRAPH,TRANS] --train [train_file] --save [model_file] --dropContext [feature_to_drop]

# prediction
imsnpars/main.py --parser [GRAPH,TRANS] --model [model_file] --test  [test_file] --output [output_file] --dropContext [feature_to_drop]
```

Possibile positions to drop:
1. Graph-based parser:
  * h -- head
  * d -- dependent
  * s -- (one random) sibling
  * c -- (one random) child of the dependent
  * h+-1_ -- (one random) word at positions +- 1 from the head which are not structurally related
  * d+-1_ -- (one random) word at positions +- 1 from the dependent which are not structurally related
  * h+-2_ -- (one random) word at positions +- 2 from the head which are not structurally related
  * d+-2_ -- (one random) word at positions +- 2 from the dependent which are not structurally related
  
2. Transition-based parser:
  * s0/s1/s2 -- first/second/third item on the stack
  * b0/b1/b2 -- first/second/third item in the buffer
  * s0L/s1L/s2L -- the left-most child of the first/second/third item on the stack
  * s0R/s1R/s2R -- the right-most child of the first/second/third item on the stack
  * s0L_/s1L_/s2L_ -- (one-random) left child of the first/second/third item on the stack which is not the left-most
  * s0R_/s1R_/s2R_ -- (one-random) right child of the first/second/third item on the stack which is not the right-most

### Tests

*Analysis tools* come with four testing scripts to check if everything works fine:
1. systests/test_surf_analysis.sh -- runs surface analysis with a small fake model
2. systests/test_feat_analysis.sh -- runs feature analysis with a small fake model
3. systests/test_drop_trans_parser.sh -- trains multiple transition-based parsers with different positions dropped
4. systests/test_drop_graph_parser.sh.sh -- trains multiple graph-based parsers with different positions dropped


