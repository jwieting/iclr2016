from ppdb_utils import getWordmap
from params import params
from ppdb_utils import getPPDBData
from ppdb_lstm_model import ppdb_lstm_model
from ppdb_rnn_model import ppdb_rnn_model
from ppdb_word_model import ppdb_word_model
from ppdb_proj_model import ppdb_proj_model
from ppdb_dan_model import ppdb_dan_model
import lasagne
import random
import numpy as np
import sys
import argparse
import ppdb_utils

def str2bool(v):
    if v is None:
        return False
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise ValueError('A type that was supposed to be boolean is not boolean.')

def learner2bool(v):
    if v is None:
        return lasagne.updates.adagrad
    if v.lower() == "adagrad":
        return lasagne.updates.adagrad
    if v.lower() == "adam":
        return lasagne.updates.adam
    raise ValueError('A type that was supposed to be a learner is not.')

random.seed(1)
np.random.seed(1)

params = params()

parser = argparse.ArgumentParser()
parser.add_argument("-LW", help="Lambda for word embeddings (normal training).", type=float)
parser.add_argument("-LC", help="Lambda for composition parameters (normal training).", type=float)
parser.add_argument("-outfile", help="Output file name.")
parser.add_argument("-batchsize", help="Size of batch.", type=int)
parser.add_argument("-dim", help="Size of input.", type=int)
parser.add_argument("-wordfile", help="Word embedding file.")
parser.add_argument("-layersize", help="Size of output layers in models.", type=int)
parser.add_argument("-updatewords", help="Whether to update the word embeddings")
parser.add_argument("-wordstem", help="Nickname of word embeddings used.")
parser.add_argument("-save", help="Whether to pickle the model.")
parser.add_argument("-train", help="Training data file.")
parser.add_argument("-margin", help="Margin in objective function.", type=float)
parser.add_argument("-samplingtype", help="Type of sampling used.")
parser.add_argument("-peephole", help="Whether to use peephole connections in LSTM.")
parser.add_argument("-outgate", help="Whether to use output gate in LSTM.")
parser.add_argument("-nonlinearity", help="Type of nonlinearity in projection and DAN model.",
                    type=int)
parser.add_argument("-nntype", help="Type of neural network.")
parser.add_argument("-evaluate", help="Whether to evaluate the model during training.")
parser.add_argument("-epochs", help="Number of epochs in training.", type=int)
parser.add_argument("-clip", help="Threshold for gradient clipping.",type=int)
parser.add_argument("-eta", help="Learning rate.", type=float)
parser.add_argument("-learner", help="Either AdaGrad or Adam.")
parser.add_argument("-add_rnn", help="Whether to keep RNN close to an addition model.")
parser.add_argument("-numlayers", help="Number of layers in DAN Model.", type=int)
parser.add_argument("-num_examples", help="Number of examples to use in training. If not set, will use all examples.", type=int)

args = parser.parse_args()

params.LW = args.LW
params.LC = args.LC
params.outfile = args.outfile
params.batchsize = args.batchsize
params.hiddensize = args.dim
params.wordfile = args.wordfile
params.nntype = args.nntype
params.layersize = args.layersize
params.updatewords = str2bool(args.updatewords)
params.wordstem = args.wordstem
params.save = str2bool(args.save)
params.train = args.train
params.margin = args.margin
params.type = args.samplingtype
params.peephole = str2bool(args.peephole)
params.outgate = str2bool(args.outgate)
params.nntype = args.nntype
params.epochs = args.epochs
params.evaluate = str2bool(args.evaluate)
params.learner = learner2bool(args.learner)
params.add_rnn = str2bool(args.add_rnn)
params.numlayers = args.numlayers

if args.eta:
    params.eta = args.eta

params.clip = args.clip
if args.clip:
    if params.clip == 0:
        params.clip = None

if args.nonlinearity:
    if args.nonlinearity == 1:
        params.nonlinearity = lasagne.nonlinearities.linear
    if args.nonlinearity == 2:
        params.nonlinearity = lasagne.nonlinearities.tanh
    if args.nonlinearity == 3:
        params.nonlinearity = lasagne.nonlinearities.rectify
    if args.nonlinearity == 4:
        params.nonlinearity = lasagne.nonlinearities.sigmoid

(words, We) = getWordmap(params.wordfile)
examples = getPPDBData(params.train, words)

if args.num_examples:
    examples = examples[0:args.num_examples]

print "Saving to: " + params.outfile

model = None

print sys.argv

if params.nntype == 'lstm':
    model = ppdb_lstm_model(We, params)
elif params.nntype == 'word':
    model = ppdb_word_model(We, params)
elif params.nntype == 'proj':
    model = ppdb_proj_model(We, params)
elif params.nntype == 'rnn':
    model = ppdb_rnn_model(We, params)
elif params.nntype == 'dan':
    model = ppdb_dan_model(We, params)
else:
    "Error no type specified"

ppdb_utils.train(model, examples, words, params)
