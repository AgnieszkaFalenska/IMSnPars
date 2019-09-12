# IMSnPars

IMS Neural Dependency Parser is a re-implementation of the transition- and graph-based parsers described in [Simple and Accurate Dependency Parsing
Using Bidirectional LSTM Feature Representations](https://aclweb.org/anthology/Q16-1023)

## Required software

> Python 3.7

> [Dynet 2.0](http://dynet.io/)

> [NetworkX package](https://networkx.github.io/)

## Usage

### Transition-based parser

Training a new model:
```
python3 imsnpar/main.py --parser TRANS --train [train_file] --save [model_file]
```

Loading and predicting with the trained model:
```
python3 imsnpar/main.py --parser TRANS --model [model_file] --test  [test_file] --output [output_file]
```

The parser supports many other options. All of them can be seen after running:
```
python3 imsnpar/main.py --parser TRANS --help
```

### Graph-based parser

Training a new model:
```
python3 imsnpar/main.py --parser GRAPH --train [train_file] --save [model_file]
```

Loading and predicting with the trained model:
```
python3 imsnpar/main.py --parser GRAPH --model [model_file] --test  [test_file] --output [output_file]
```

The parser supports many other options. All of them can be seen after running:
```
python3 imsnpar/main.py --parser GRAPH --help
```

### Tests

*IMSnPars* comes with four testing scripts to check if everything works fine:
1. systests/test_trans_parser.sh -- trains a new transition-based parser on small fake data and uses this model for prediction
2. systests/test_graph_parser.sh -- trains a new graph-based parser on small fake data and uses this model for prediction
3. systests/test_all_trans_parsers.sh -- trains multiple transition-based models with different sets of options
4. systests/test_all_graph_parsers.sh -- trains multiple graph-based models with different sets of options

We recommend running the two first scripts before using *IMSnPars* for other purposes (both tests take less than a minute). Both of the scripts should end with an information that everything went fine. Transition-based parser achieves LAS=64.61 on the fake data and the graph-based one LAS=66.47.
