# PyTorch PoS Tagging [In Progress]

This repo contains tutorials covering how to do part-of-speech (PoS) tagging using [PyTorch](https://github.com/pytorch/pytorch) 1.2 and [TorchText](https://github.com/pytorch/text) 0.4 using Python 3.7.

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

* 1 - [Simple RNN POS Tagger](https://github.com/bentrevett/pytorch-pos-tagging/blob/master/1%20-%20Simple%20RNN%20PoS%20Tagger.ipynb)

    This tutorial covers how to implement the most basic of PoS models - a multi-layer bi-directional RNN with pre-trained  GloVe embeddings. 

* 2 - [NestedField, CharCNN and Inference](https://github.com/bentrevett/pytorch-pos-tagging/blob/master/2%20-%20NestedField%2C%20CharCNN%20and%20Inference.ipynb)

    Now we have a basic PoS tagger working we can improve on it. In this tutorial we introduce the `NestedField` - a TorchText field that processes another field. The `NestedField` provides an easy way to get both the words and characters for the sequences we want to tag. We continue to embed the words as before, using an embedding layer, but we embed the characters using a convolutional neural network (CNN). Finally, we show how to use the model for inference, allowing us to tag any input sentence.

## References

Here are some things I looked at while making these tutorials. Some of it may be out of date.

- https://github.com/pytorch/text/blob/master/torchtext/datasets/sequence_tagging.py
- https://github.com/pytorch/text/blob/master/test/sequence_tagging.py