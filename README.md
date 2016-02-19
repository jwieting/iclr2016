# iclr2016

Code to train models from "Towards Universal Paraphrastic Sentence Embeddings".

The code is written in python and requires numpy, scipy, theano and the lasagne library.

To get started, run setup.sh to download initial word embeddings and PPDB training data. There is a demo script that takes the model that you would like to train as a command line argument (check the script to see available choices). Check main/ppdb_train.py and main/train.py for command line options.

The code is separated into 3 parts:

* similarity: contains code for training models on the SICK similarity and entailment tasks
* main: contains code for training models on PPDB data as well as various utilities
* sentiment: contains code for training sentiment models.

If you use our code for your work please cite:

@article{wieting2016iclr,
author    = {John Wieting and Mohit Bansal and Kevin Gimpel and Karen Livescu},
title     = {Towards Universal Paraphrastic Sentence Embeddings},
journal   = {CoRR},
volume    = {abs/1511.08198},
year      = {2015}}