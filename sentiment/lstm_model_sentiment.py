import theano
import numpy as np
from theano import tensor as T
from theano import config
from lasagne_lstm_nooutput import lasagne_lstm_nooutput
import lasagne
import cPickle

class lstm_model_sentiment(object):

    def getRegTerm(self, params, We, initial_We, l_out, l_softmax, pickled_params):
        if params.traintype == "normal":
            l2 = 0.5*params.LC*sum(lasagne.regularization.l2(x) for x in self.network_params)
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
            print p
            #contains [<TensorType(float64, matrix)>,
            # W_in_to_ingate, W_hid_to_ingate, b_ingate, W_in_to_forgetgate,
            # W_hid_to_forgetgate, b_forgetgate, W_in_to_cell, W_hid_to_cell,
            # b_cell, W_in_to_outgate, W_hid_to_outgate, b_outgate, W_cell_to_ingate,
            # W_cell_to_forgetgate, W_cell_to_outgate]

        if params.traintype == "reg":
            print "regularizing to parameters"

        if params.traintype == "rep":
            print "not updating embeddings"

        #params
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

        l_in = lasagne.layers.InputLayer((None, None, 1))
        l_mask = lasagne.layers.InputLayer(shape=(None, None))
        l_emb = lasagne.layers.EmbeddingLayer(l_in, input_size=We.get_value().shape[0], output_size=We.get_value().shape[1], W=We)
        l_lstm = None
        if params.useoutgate:
            l_lstm = lasagne.layers.LSTMLayer(l_emb, params.layersize, peepholes=params.usepeep, learn_init=False, mask_input = l_mask)
        else:
            l_lstm = lasagne_lstm_nooutput(l_emb, params.layersize, peepholes=params.usepeep, learn_init=False, mask_input = l_mask)

        if params.traintype == "reg" or params.traintype == "rep":
            if params.useoutgate:
                W_in_to_ingate = np.asarray(p[1].get_value(), dtype = config.floatX)
                W_hid_to_ingate = np.asarray(p[2].get_value(), dtype = config.floatX)
                b_ingate = np.asarray(p[3].get_value(), dtype = config.floatX)
                W_in_to_forgetgate = np.asarray(p[4].get_value(), dtype = config.floatX)
                W_hid_to_forgetgate = np.asarray(p[5].get_value(), dtype = config.floatX)
                b_forgetgate = np.asarray(p[6].get_value(), dtype = config.floatX)
                W_in_to_cell = np.asarray(p[7].get_value(), dtype = config.floatX)
                W_hid_to_cell = np.asarray(p[8].get_value(), dtype = config.floatX)
                b_cell = np.asarray(p[9].get_value(), dtype = config.floatX)
                W_in_to_outgate = np.asarray(p[10].get_value(), dtype = config.floatX)
                W_hid_to_outgate = np.asarray(p[11].get_value(), dtype = config.floatX)
                b_outgate = np.asarray(p[12].get_value(), dtype = config.floatX)
                W_cell_to_ingate = np.asarray(p[13].get_value(), dtype = config.floatX)
                W_cell_to_forgetgate = np.asarray(p[14].get_value(), dtype = config.floatX)
                W_cell_to_outgate = np.asarray(p[15].get_value(), dtype = config.floatX)

                ingate = lasagne.layers.Gate(W_in=W_in_to_ingate, W_hid=W_hid_to_ingate, W_cell=W_cell_to_ingate, b=b_ingate)
                forgetgate = lasagne.layers.Gate(W_in=W_in_to_forgetgate, W_hid=W_hid_to_forgetgate, W_cell=W_cell_to_forgetgate, b=b_forgetgate)
                outgate = lasagne.layers.Gate(W_in=W_in_to_outgate, W_hid=W_hid_to_outgate, W_cell=W_cell_to_outgate, b=b_outgate)
                cell = lasagne.layers.Gate(W_in=W_in_to_cell, W_hid=W_hid_to_cell, W_cell=None, b=b_cell, nonlinearity=lasagne.nonlinearities.tanh)
                l_lstm = lasagne.layers.LSTMLayer(l_emb, params.layersize, ingate = ingate, forgetgate = forgetgate,
                                  outgate = outgate, cell = cell, peepholes=params.usepeep, learn_init=False, mask_input = l_mask)
            else:
                W_in_to_ingate = np.asarray(p[1].get_value(), dtype = config.floatX)
                W_hid_to_ingate = np.asarray(p[2].get_value(), dtype = config.floatX)
                b_ingate = np.asarray(p[3].get_value(), dtype = config.floatX)
                W_in_to_forgetgate = np.asarray(p[4].get_value(), dtype = config.floatX)
                W_hid_to_forgetgate = np.asarray(p[5].get_value(), dtype = config.floatX)
                b_forgetgate = np.asarray(p[6].get_value(), dtype = config.floatX)
                W_in_to_cell = np.asarray(p[7].get_value(), dtype = config.floatX)
                W_hid_to_cell = np.asarray(p[8].get_value(), dtype = config.floatX)
                b_cell = np.asarray(p[9].get_value(), dtype = config.floatX)
                W_cell_to_ingate = np.asarray(p[10].get_value(), dtype = config.floatX)
                W_cell_to_forgetgate = np.asarray(p[11].get_value(), dtype = config.floatX)

                ingate = lasagne.layers.Gate(W_in=W_in_to_ingate, W_hid=W_hid_to_ingate, W_cell=W_cell_to_ingate, b=b_ingate)
                forgetgate = lasagne.layers.Gate(W_in=W_in_to_forgetgate, W_hid=W_hid_to_forgetgate, W_cell=W_cell_to_forgetgate, b=b_forgetgate)
                cell = lasagne.layers.Gate(W_in=W_in_to_cell, W_hid=W_hid_to_cell, W_cell=None, b=b_cell, nonlinearity=lasagne.nonlinearities.tanh)
                l_lstm = lasagne_lstm_nooutput(l_emb, params.layersize, ingate = ingate, forgetgate = forgetgate,
                                  cell = cell, peepholes=params.usepeep, learn_init=False, mask_input = l_mask)

        l_out = lasagne.layers.SliceLayer(l_lstm, -1, 1)

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
