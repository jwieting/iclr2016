import theano
import numpy as np
from theano import tensor as T
from theano import config
import lasagne
from lasagne_average_layer import lasagne_average_layer
import cPickle

class word_model_sentiment(object):

    def getRegTerm(self, params, We, initial_We):
        l2 = 0.5*params.LC*sum(lasagne.regularization.l2(x) for x in self.network_params)
        if params.traintype == "normal":
            if params.updatewords:
                return l2 + 0.5*params.LW*lasagne.regularization.l2(We-initial_We)
            else:
                return l2
        elif params.traintype == "reg":
            return l2 + 0.5*params.LRW*lasagne.regularization.l2(We-initial_We)
        elif params.traintype == "rep":
            return l2
        else:
            raise ValueError('Params.traintype not set correctly.')

    def getTrainableParams(self, params):
        if params.traintype == "rep":
            return self.network_params
        if params.updatewords or params.traintype == "reg":
            return self.all_params
        else:
            return self.network_params

    def __init__(self, We_initial, params):

        p = None

        if params.traintype == "reg" or params.traintype == "rep":
            p = cPickle.load(file(params.regfile, 'rb'))
            print p #containes We

        if params.traintype == "reg":
            print "regularizing to parameters"

        if params.traintype == "rep":
            print "not updating embeddings"

        initial_We = theano.shared(np.asarray(We_initial, dtype = config.floatX))
        We = theano.shared(np.asarray(We_initial, dtype = config.floatX))

        if params.traintype == "reg":
            initial_We = theano.shared(np.asarray(p[0].get_value(), dtype = config.floatX))
            We = theano.shared(np.asarray(p[0].get_value(), dtype = config.floatX))

        if params.traintype == "rep":
            We = theano.shared(np.asarray(p[0].get_value(), dtype = config.floatX))

        g1batchindices = T.imatrix()
        g1mask = T.matrix()
        scores = T.matrix()

        l_in = lasagne.layers.InputLayer((None, None, 1))
        l_mask = lasagne.layers.InputLayer(shape=(None, None))
        l_emb = lasagne.layers.EmbeddingLayer(l_in, input_size=We.get_value().shape[0], output_size=We.get_value().shape[1], W=We)
        l_out = lasagne_average_layer([l_emb, l_mask])

        embg = lasagne.layers.get_output(l_out, {l_in:g1batchindices, l_mask:g1mask})

        l_in2 = lasagne.layers.InputLayer((None, We.get_value().shape[1]))
        l_sigmoid = lasagne.layers.DenseLayer(l_in2, params.memsize, nonlinearity=lasagne.nonlinearities.sigmoid)

        l_softmax = lasagne.layers.DenseLayer(l_sigmoid, 2, nonlinearity=T.nnet.softmax)
        X = lasagne.layers.get_output(l_softmax, {l_in2:embg})
        cost = T.nnet.categorical_crossentropy(X,scores)
        prediction = T.argmax(X, axis=1)

        self.network_params = lasagne.layers.get_all_params(l_out, trainable=True) + lasagne.layers.get_all_params(l_softmax, trainable=True)
        self.network_params.pop(0)
        self.all_params = lasagne.layers.get_all_params(l_out, trainable=True) + lasagne.layers.get_all_params(l_softmax, trainable=True)

        reg = self.getRegTerm(params, We, initial_We)
        self.trainable = self.getTrainableParams(params)
        cost = T.mean(cost) + reg

        self.feedforward_function = theano.function([g1batchindices,g1mask], embg)
        self.scoring_function = theano.function([g1batchindices,
                             g1mask],prediction)
        self.cost_function = theano.function([scores, g1batchindices,
                             g1mask], cost)

        grads = theano.gradient.grad(cost, self.trainable)
        if params.clip:
            grads = [lasagne.updates.norm_constraint(grad, params.clip, range(grad.ndim)) for grad in grads]
        updates = params.learner(grads, self.trainable, params.eta)
        self.train_function = theano.function([scores, g1batchindices,
                             g1mask], cost, updates=updates)
