# IMSnPars

IMS Neural Dependency Parser is a re-implementation of the transition- and graph-based parsers described in [Simple and Accurate Dependency Parsing
Using Bidirectional LSTM Feature Representations](https://aclweb.org/anthology/Q16-1023)

## Required software

```
Python 3.7
[Dynet 2.0](http://dynet.io/)
[NetworkX package](https://networkx.github.io/)
```

## Usage

### Transition-based parser

Training a new model:
```
python3 src/main.py --parser TRANS --train [train_file] --save [model_file]
```

Loading and predicting with the trained model:
```
python3 src/main.py --parser TRANS --model [model_file] --test  [test_file] --output [output_file]
```

The parser supports many other options. All of them can be seen after running:
```
python3 src/main.py --parser TRANS --help
```

### Graph-based parser

Training a new model:
```
python3 src/main.py --parser GRAPH --train [train_file] --save [model_file]
```

Loading and predicting with the trained model:
```
python3 src/main.py --parser GRAPH --model [model_file] --test  [test_file] --output [output_file]
```

The parser supports many other options. All of them can be seen after running:
```
python3 src/main.py --parser GRAPH --help
```
