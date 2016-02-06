import theano
import numpy as np
from theano import tensor as T
from theano import config
from lasagne_average_layer import lasagne_average_layer
import lasagne
import cPickle

class dan_model_sentiment(object):

    def getRegTerm(self, params, We, initial_We, l_out, l_softmax, pickled_params):
        if params.traintype == "normal":
            l2 = 0.5*params.LC*sum(lasagne.regularization.l2(x) for x in self.network_params)
            if params.updatewords:
                return l2 + 0.5*params.LW*lasagne.regularization.l2(We-initial_We)
            else:
                return l2
        elif params.traintype == "reg":
            tmp = lasagne.layers.get_all_params(l_out, trainable=True)
            l2 = 0.5*params.LRC*(lasagne.regularization.l2(tmp[1]-np.asarray(pickled_params[1].get_value(), dtype = config.floatX)))
            l2 += 0.5*params.LRC*(lasagne.regularization.l2(tmp[2]-np.asarray(pickled_params[2].get_value(), dtype = config.floatX)))

            tmp = lasagne.layers.get_all_params(l_softmax, trainable=True)
            l2 += 0.5*params.LC*sum(lasagne.regularization.l2(x) for x in tmp)

            return l2 + 0.5*params.LRW*lasagne.regularization.l2(We-initial_We)
        elif params.traintype == "rep":
            tmp = lasagne.layers.get_all_params(l_softmax, trainable=True)
            l2 = 0.5*params.LC*sum(lasagne.regularization.l2(x) for x in self.network_params)
            return l2
        else:
            raise ValueError('params.traintype not set correctly.')

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
            print p #containes We, W, and b

        if params.traintype == "reg":
            print "regularizing to parameters"
            raise NotImplementedError("Not implemented for DAN model.")

        if params.traintype == "rep":
            print "not updating embeddings"
            raise NotImplementedError("Not implemented for DAN model.")

        initial_We = theano.shared(np.asarray(We_initial, dtype = config.floatX))
        We = theano.shared(np.asarray(We_initial, dtype = config.floatX))

        if params.traintype == "reg":
            initial_We = theano.shared(np.asarray(p[0].get_value(), dtype = config.floatX))
            We = theano.shared(np.asarray(p[0].get_value(), dtype = config.floatX))
            updatewords = True

        if params.traintype == "rep":
            We = theano.shared(np.asarray(p[0].get_value(), dtype = config.floatX))
            updatewords = False

        g1batchindices = T.imatrix()
        g1mask = T.matrix()
        scores = T.matrix()

        if params.traintype == "reg" or params.traintype == "rep":
            W = np.asarray(p[1].get_value(), dtype = config.floatX)
            b = np.asarray(p[2].get_value(), dtype = config.floatX)

        l_in = lasagne.layers.InputLayer((None, None))
        l_mask = lasagne.layers.InputLayer(shape=(None, None))
        l_emb = lasagne.layers.EmbeddingLayer(l_in, input_size=We.get_value().shape[0], output_size=We.get_value().shape[1], W=We)
        l_average = lasagne_average_layer([l_emb, l_mask])
        l_out = lasagne.layers.DenseLayer(l_average, params.layersize, nonlinearity=params.nonlinearity)

        if params.traintype == "reg" or params.traintype == "rep":
            l_in = lasagne.layers.InputLayer((None, None))
            l_mask = lasagne.layers.InputLayer(shape=(None, None))
            l_emb = lasagne.layers.EmbeddingLayer(l_in, input_size=We.get_value().shape[0], output_size=We.get_value().shape[1], W=We)
            l_average = lasagne_average_layer([l_emb, l_mask])
            l_out = lasagne.layers.DenseLayer(l_average, params.layersize, nonlinearity=params.nonlinearity,W=W, b=b)

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

        reg = self.getRegTerm(params, We, initial_We, l_out, l_softmax, p)
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
