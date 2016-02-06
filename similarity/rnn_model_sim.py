import theano
import numpy as np
from theano import tensor as T
from theano import config
import lasagne
import cPickle

class rnn_model_sim(object):

    def getRegTerm(self, params, We, initial_We, l_out, l_softmax, pickled_params):
        if params.traintype == "normal" and not params.add_rnn:
            l2 = 0.5*params.LC*sum(lasagne.regularization.l2(x) for x in self.network_params)
            if params.updatewords:
                return l2 + 0.5*params.LW*lasagne.regularization.l2(We-initial_We)
            else:
                return l2
        elif params.traintype == "normal" and params.add_rnn:
            identity_mat = np.asarray(np.eye(params.layersize), dtype = config.floatX)
            rnn_params = lasagne.layers.get_all_params(l_out, trainable=True)
            rnn_params.pop(0)
            l2 = 0.5 * params.LC * lasagne.regularization.l2(rnn_params[0] - identity_mat)
            l2 += 0.5 * params.LC * lasagne.regularization.l2(rnn_params[2] - identity_mat)
            l2 += 0.5 * params.LC * lasagne.regularization.l2(rnn_params[1])
            classify_params = lasagne.layers.get_all_params(l_softmax, trainable=True)
            l2 += 0.5*params.LC*sum(lasagne.regularization.l2(x) for x in classify_params)
            if params.updatewords:
                return l2 + 0.5*params.LW*lasagne.regularization.l2(We-initial_We)
            else:
                return l2
        elif params.traintype == "reg":
            tmp = lasagne.layers.get_all_params(l_out, trainable=True)
            idx = 1
            l2 = 0.
            while idx < len(tmp):
                l2 += 0.5*params.LRC*(lasagne.regularization.l2(tmp[idx]-np.asarray(pickled_params[idx].get_value(), dtype = config.floatX)))
                idx += 1
            tmp = lasagne.layers.get_all_params(l_softmax, trainable=True)
            l2 += 0.5*params.LC*sum(lasagne.regularization.l2(x) for x in tmp)
            return l2 + 0.5*params.LRW*lasagne.regularization.l2(We-initial_We)
        elif params.traintype == "rep":
            tmp = lasagne.layers.get_all_params(l_softmax, trainable=True)
            l2 = 0.5*params.LC*sum(lasagne.regularization.l2(x) for x in tmp)
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

        if params.maxval:
            self.nout = params.maxval - params.minval + 1

        p = None
        if params.traintype == "reg" or params.traintype == "rep":
            p = cPickle.load(file(params.regfile, 'rb'))
            print p
            #contains [<TensorType(float64, matrix)>, W, b, W

        if params.traintype == "reg":
            print "regularizing to parameters"

        if params.traintype == "rep":
            print "not updating embeddings"

        initial_We = theano.shared(np.asarray(We_initial, dtype = config.floatX))
        We = theano.shared(np.asarray(We_initial, dtype = config.floatX))

        if params.traintype == "reg":
            initial_We = theano.shared(np.asarray(p[0].get_value(), dtype = config.floatX))
            We = theano.shared(np.asarray(p[0].get_value(), dtype = config.floatX))
            updatewords = True

        if params.traintype == "rep":
            We = theano.shared(np.asarray(p[0].get_value(), dtype = config.floatX))
            updatewords = False

        g1batchindices = T.imatrix(); g2batchindices = T.imatrix()
        g1mask = T.matrix(); g2mask = T.matrix()
        scores = T.matrix()

        l_in = lasagne.layers.InputLayer((None, None, 1))
        l_mask = lasagne.layers.InputLayer(shape=(None, None))
        l_emb = lasagne.layers.EmbeddingLayer(l_in, input_size=We.get_value().shape[0], output_size=We.get_value().shape[1], W=We)
        l_rnn = None
        if params.add_rnn:
            identity_mat = np.asarray(np.eye(params.layersize), dtype = config.floatX)
            l_rnn = lasagne.layers.RecurrentLayer(l_emb, params.layersize, W_in_to_hid=identity_mat,
                                                  W_hid_to_hid=identity_mat, nonlinearity=lasagne.nonlinearities.linear,
                                                  learn_init=False, mask_input=l_mask)
        else:
            l_rnn = lasagne.layers.RecurrentLayer(l_emb, params.layersize, nonlinearity=params.nonlinearity,
                                                  learn_init=False, mask_input=l_mask)

        if params.traintype == "reg" or params.traintype == "rep":
            W1 = np.asarray(p[1].get_value(), dtype = config.floatX)
            b = np.asarray(p[2].get_value(), dtype = config.floatX)
            W2 = np.asarray(p[3].get_value(), dtype = config.floatX)
            if params.add_rnn:
                l_rnn = lasagne.layers.RecurrentLayer(l_emb, params.layersize, W_in_to_hid=W1,
                                                  W_hid_to_hid=W2, b=b, nonlinearity=lasagne.nonlinearities.linear,
                                                  learn_init=False, mask_input=l_mask)

            else:
                l_rnn = lasagne.layers.RecurrentLayer(l_emb, params.layersize, W_in_to_hid=W1,
                                                  W_hid_to_hid=W2, b=b, nonlinearity=params.nonlinearity,
                                                  learn_init=False, mask_input=l_mask)

        l_out = lasagne.layers.SliceLayer(l_rnn, -1, 1)

        embg1 = lasagne.layers.get_output(l_out, {l_in:g1batchindices, l_mask:g1mask})
        embg2 = lasagne.layers.get_output(l_out, {l_in:g2batchindices, l_mask:g2mask})

        if params.add_rnn:
            embg1 = embg1 / g1mask.sum(axis=1)[:, None]
            embg2 = embg2 / g2mask.sum(axis=1)[:, None]

        g1_dot_g2 = embg1*embg2
        g1_abs_g2 = abs(embg1-embg2)

        lin_dot = lasagne.layers.InputLayer((None, params.layersize))
        lin_abs = lasagne.layers.InputLayer((None, params.layersize))
        l_sum = lasagne.layers.ConcatLayer([lin_dot, lin_abs])
        l_sigmoid = lasagne.layers.DenseLayer(l_sum, params.memsize, nonlinearity=lasagne.nonlinearities.sigmoid)

        if params.task == "sim":
            l_softmax = lasagne.layers.DenseLayer(l_sigmoid, self.nout, nonlinearity=T.nnet.softmax)
            X = lasagne.layers.get_output(l_softmax, {lin_dot:g1_dot_g2, lin_abs:g1_abs_g2})
            Y = T.log(X)

            cost = scores*(T.log(scores) - Y)
            cost = cost.sum(axis=1)/(float(self.nout))

            prediction = 0.
            i = params.minval
            while i<= params.maxval:
                prediction = prediction + i*X[:,i-1]
                i += 1
        elif params.task == "ent":
            l_softmax = lasagne.layers.DenseLayer(l_sigmoid, 3, nonlinearity=T.nnet.softmax)
            X = lasagne.layers.get_output(l_softmax, {lin_dot:g1_dot_g2, lin_abs:g1_abs_g2})

            cost = theano.tensor.nnet.categorical_crossentropy(X,scores)

            prediction = T.argmax(X, axis=1)
        else:
            raise ValueError('Params.task not set correctly.')

        self.network_params = lasagne.layers.get_all_params(l_out, trainable=True) + lasagne.layers.get_all_params(l_softmax, trainable=True)
        self.network_params.pop(0)
        self.all_params = lasagne.layers.get_all_params(l_out, trainable=True) + lasagne.layers.get_all_params(l_softmax, trainable=True)

        reg = self.getRegTerm(params, We, initial_We, l_out, l_softmax, p)
        self.trainable = self.getTrainableParams(params)
        cost = T.mean(cost) + reg

        self.feedforward_function = theano.function([g1batchindices,g1mask], embg1)
        self.scoring_function = theano.function([g1batchindices, g2batchindices,
                             g1mask, g2mask],prediction)
        self.cost_function = theano.function([scores, g1batchindices, g2batchindices,
                             g1mask, g2mask], cost)

        grads = theano.gradient.grad(cost, self.trainable)
        if params.clip:
            grads = [lasagne.updates.norm_constraint(grad, params.clip, range(grad.ndim)) for grad in grads]
        updates = params.learner(grads, self.trainable, params.eta)
        self.train_function = theano.function([scores, g1batchindices, g2batchindices,
                             g1mask, g2mask], cost, updates=updates)