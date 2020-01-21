# PyTorch PoS Tagging [In Progress]

This repo contains tutorials covering how to do part-of-speech (PoS) tagging using [PyTorch](https://github.com/pytorch/pytorch) 1.4 and [TorchText](https://github.com/pytorch/text) 0.5 using Python 3.7.

These tutorials will cover getting started with the de facto approach to PoS tagging: recurrent neural networks (RNNs). The first introduces a simple RNN network. The second covers how to use TorchText's `NestedField` in order to get the characters for each word, and how to feed these to a CharCNN. 

**If you find any mistakes or disagree with any of the explanations, please do not hesitate to [submit an issue](https://github.com/bentrevett/pytorch-pos-tagging/issues/new). I welcome any feedback, positive or negative!**

## Getting Started

To install PyTorch, see installation instructions on the [PyTorch website](pytorch.org).

To install TorchText:

``` bash
pip install torchtext
```

We'll also make use of spaCy to tokenize our data. To install spaCy, follow the instructions [here](https://spacy.io/usage/) making sure to install the English models:

``` bash
python -m spacy download en
```

## Tutorials

* 1 - [BiLSTM POS Tagger](https://github.com/bentrevett/pytorch-pos-tagging/blob/master/1%20-%20BiLSTM%20PoS%20Tagger.ipynb)

    This tutorial covers the workflow of a PoS tagging project with PyTorch and TorchText. We'll introduce the basic TorchText concepts such as: defining how data is processed; using TorchText's datasets and how to use pre-trained embeddings. Using PyTorch we built a strong baseline model: a multi-layer bi-directional LSTM. We also show how the model can be used for inference to tag any input text.

## References

Here are some things I looked at while making these tutorials. Some of it may be out of date.

- https://github.com/pytorch/text/blob/master/torchtext/datasets/sequence_tagging.py
- https://github.com/pytorch/text/blob/master/test/sequence_tagging.py