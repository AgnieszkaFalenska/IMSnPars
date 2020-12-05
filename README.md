# IMSnPars

IMS Neural Dependency Parser is a re-implementation of the transition- and graph-based parsers described in [Simple and Accurate Dependency Parsing
Using Bidirectional LSTM Feature Representations](https://aclweb.org/anthology/Q16-1023)

The parser was developed for the paper [The (Non-)Utility of Structural Features in BiLSTM-based
Dependency Parsers](https://www.aclweb.org/anthology/P19-1012) (see [acl2019 branch](https://github.com/AgnieszkaFalenska/IMSnPars/tree/acl2019) for all the paper specific changes and analysis tools):

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

Later it was extended with multi-task training for the paper [Integrating Graph-Based and Transition-Based Dependency Parsers in the Deep Contextualized Era](https://www.aclweb.org/anthology/2020.iwpt-1.4.pdf) (see [iwpt2020 branch](https://github.com/AgnieszkaFalenska/IMSnPars/tree/iwpt2020) for all the paper specific changes and analysis tools):

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

## Usage

### Installation 
We suggest to make use of python's [virtual environments](https://docs.python.org/3/tutorial/venv.html) for your project.

```sh
# install virtual env 
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# install imsnpars package
pip install -r requirements.txt --use-feature=2020-resolver
pip install -r requirements-dev.txt --use-feature=2020-resolver
python setup.py develop -q

# download test data and serialized models
imsnpars_downloader.py --systests
imsnpars_downloader.py --hdt
```

### Training a new Model
There are two types of dependency parsers available
that need to be specified with the `--parser` flag:

- **Transition-based parser** (`TRANS`)
- **Graph-based parser** (`GRAPH`)

The training set must be `.conllu` file, 
and its path is specified with the `--train` flag.
Further a file path must be specified with the `--save` flag
where to store the trained and serialized model.

```sh
python3 imsnpars/main.py \
    --parser [TRANS or GRAPH] \
    --train [train_file] \
    --save [model_file]
```

Example, given you downloaded the [HDT treebank dataset](https://github.com/UniversalDependencies/UD_German-HDT/releases/tag/r2.7) mentionend above (run `imsnpars_downloader.py --hdt`),
we train a new model with the transition-based parser (TRANS):

```sh
mkdir -p "${HOME}/imsnpars_data/my-new-model-v1.2.3"

python3 imsnpars/main.py \
    --parser TRANS \
    --train="${HOME}/imsnpars_data/hdt/train.conllu"  \
    --save="${HOME}/imsnpars_data/my-new-model-v1.2.3"
```

### Inference with a Pre-trained Model
For evaluation as well as production purposes, 
we can load a pre-trained model as explained in the [previous chapter](#training-a-new-model).
Both input data (`--test`) and output data (`--output`) are `.conllu` files.

```sh
python3 imsnpars/main.py \
    --parser [TRANS or GRAPH] \
    --model [model_file] \
    --test [test_file] \
    --output [output_file]
```

Analogous to the previous example,
we can run inference on the HDT test set with our pre-trained model:

```sh
mkdir -p "${HOME}/imsnpars_data/output"

python3 imsnpars/main.py \
    --parser TRANS \
    --model="${HOME}/imsnpars_data/my-new-model-v1.2.3" \
    --test="${HOME}/imsnpars_data/hdt/test.conllu"  \
    --output="${HOME}/imsnpars_data/output/predicted.conllu"
```

### Other settings
The parser supports many other options. All of them can be seen after running:
```sh
python3 imsnpars/main.py --parser TRANS --help
python3 imsnpars/main.py --parser GRAPH --help
```

### Tests

*IMSnPars* comes with six testing scripts to check if everything works fine:
1. `systests/test_trans_parser.sh` -- trains a new transition-based parser on small fake data and uses this model for prediction
2. `systests/test_graph_parser.sh` -- trains a new graph-based parser on small fake data and uses this model for prediction
3. `systests/test_fasttext_parser.sh` -- trains a new parser using external embeddings
4. `systests/test_elmo_parser.sh` -- trains a new parser using ELMo representations
5. `systests/test_all_trans_parsers.sh` -- trains multiple transition-based models with different sets of options
6. `systests/test_all_graph_parsers.sh` -- trains multiple graph-based models with different sets of options

Please make sure that the software is installed as python package, e.g. run `python setup.py develop -q`.

We recommend running the four first scripts before using *IMSnPars* for other purposes (all tests take less than a minute). All of the scripts should end with an information that everything went fine. For the first two tests: transition-based parser achieves LAS=64.61 on the fake data and the graph-based one LAS=66.47.
